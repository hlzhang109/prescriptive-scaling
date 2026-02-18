#!/usr/bin/env python3
"""Artifact manifest and regression diff harness.

This tool captures deterministic snapshots of generated artifacts and compares
current outputs against a baseline manifest. PNGs are compared by pixel hash
only (metadata differences are ignored). PDFs are *not* gated on binary hashes;
only presence and non-empty size are verified because PDF metadata is often
non-deterministic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


IGNORED_NAMES = {".DS_Store"}
IGNORED_DIRS = {"__pycache__"}
IGNORED_SUFFIXES = {".pyc"}


@dataclass
class Record:
    path: str
    type: str
    bytes_sha256: str
    pixel_sha256: str
    size_bytes: int

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> "Record":
        return cls(
            path=row["path"],
            type=row["type"],
            bytes_sha256=row.get("bytes_sha256", ""),
            pixel_sha256=row.get("pixel_sha256", ""),
            size_bytes=int(row.get("size_bytes", "0") or 0),
        )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _normalize_array_to_rgba(arr):
    import numpy as np

    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[2] == 3:
        alpha = np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=arr.dtype)
        arr = np.concatenate([arr, alpha], axis=2)
    if arr.shape[2] != 4:
        raise ValueError("Unsupported channel count for PNG array")

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0 + 0.5).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def _png_pixel_hash(path: Path) -> str:
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as img:
            img = img.convert("RGBA")
            width, height = img.size
            payload = img.tobytes()
    except Exception:
        payload = None
        width = height = None

    if payload is None:
        try:
            import imageio.v2 as imageio  # type: ignore
            import numpy as np  # noqa: F401

            arr = imageio.imread(path)
            arr = _normalize_array_to_rgba(arr)
            height, width, _ = arr.shape
            payload = arr.tobytes()
        except Exception:
            import matplotlib.image as mpimg  # type: ignore
            import numpy as np  # noqa: F401

            arr = mpimg.imread(path)
            arr = _normalize_array_to_rgba(arr)
            height, width, _ = arr.shape
            payload = arr.tobytes()

    if payload is None or width is None or height is None:
        return ""

    header = f"{width}x{height}:RGBA".encode("ascii")
    return _sha256_bytes(header + payload)


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")
    if root.is_file():
        yield root
        return

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted([d for d in dirnames if d not in IGNORED_DIRS])
        for filename in sorted(filenames):
            if filename in IGNORED_NAMES:
                continue
            suffix = Path(filename).suffix
            if suffix in IGNORED_SUFFIXES:
                continue
            yield Path(dirpath) / filename


def _record_for_path(path: Path, base_dir: Path) -> Record:
    size_bytes = path.stat().st_size
    try:
        rel = path.relative_to(base_dir).as_posix()
    except ValueError:
        rel = path.resolve().as_posix()
    suffix = path.suffix.lower()

    bytes_hash = _sha256_file(path)
    pixel_hash = ""
    if suffix == ".png":
        pixel_hash = _png_pixel_hash(path)

    return Record(
        path=rel,
        type="file",
        bytes_sha256=bytes_hash,
        pixel_sha256=pixel_hash,
        size_bytes=size_bytes,
    )


def _write_manifest(records: Sequence[Record], out_path: Optional[Path]) -> None:
    header = ["path", "type", "bytes_sha256", "pixel_sha256", "size_bytes"]
    out_file = sys.stdout if out_path is None else out_path.open("w", newline="")
    try:
        writer = csv.writer(out_file, delimiter="\t")
        writer.writerow(header)
        for record in records:
            writer.writerow(
                [
                    record.path,
                    record.type,
                    record.bytes_sha256,
                    record.pixel_sha256,
                    str(record.size_bytes),
                ]
            )
    finally:
        if out_path is not None:
            out_file.close()


def _load_manifest(path: Path) -> Dict[str, Record]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Manifest missing header: {path}")
        required = {"path", "type", "bytes_sha256", "pixel_sha256", "size_bytes"}
        if not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Manifest missing columns: {path}")
        records = {}
        for row in reader:
            record = Record.from_row(row)
            records[record.path] = record
        return records


def _compare_records(
    base: Record, current: Record
) -> Optional[str]:
    if base.type != current.type:
        return f"type {base.type} -> {current.type}"

    suffix = Path(base.path).suffix.lower()
    if suffix == ".png":
        if not base.pixel_sha256 or not current.pixel_sha256:
            return "missing pixel hash"
        if base.pixel_sha256 != current.pixel_sha256:
            return "pixel hash differs"
        return None

    if suffix == ".pdf":
        if current.size_bytes <= 0:
            return "pdf is empty"
        return None

    if not base.bytes_sha256 or not current.bytes_sha256:
        return "missing bytes hash"
    if base.bytes_sha256 != current.bytes_sha256:
        return "bytes hash differs"
    return None


def _write_env_snapshot(out_path: Path) -> None:
    lines: List[str] = []

    lines.append(f"python_executable: {sys.executable}")
    lines.append("python_version:")
    lines.append(_run_command([sys.executable, "-V"]))

    lines.append("pip_freeze:")
    lines.append(_run_command([sys.executable, "-m", "pip", "freeze"]))

    try:
        import matplotlib
        from matplotlib import font_manager

        lines.append(f"matplotlib_version: {matplotlib.__version__}")
        lines.append(f"matplotlib_backend: {matplotlib.get_backend()}")

        font_names = {f.name for f in font_manager.fontManager.ttflist}
        has_tnr = "Times New Roman" in font_names
        lines.append(f"font_times_new_roman_available: {has_tnr}")
        if has_tnr:
            try:
                font_path = font_manager.findfont("Times New Roman")
                lines.append(f"font_times_new_roman_path: {font_path}")
            except Exception:
                pass
    except Exception as exc:
        lines.append(f"matplotlib_info_error: {exc}")

    lines.append(f"env_PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED')}")
    lines.append(f"env_MPLBACKEND: {os.environ.get('MPLBACKEND')}")

    lines.append("git_rev_parse_HEAD:")
    lines.append(_run_command(["git", "rev-parse", "HEAD"]))
    lines.append("git_status_porcelain:")
    lines.append(_run_command(["git", "status", "--porcelain"]))

    lines.append(
        "pdf_verification: presence and non-empty only; binary hashes are ignored"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n")


def _run_command(cmd: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"error: {exc}"


def _snapshot(args: argparse.Namespace) -> int:
    base_dir = Path(args.base_dir).resolve()
    roots = [Path(root).resolve() for root in args.roots]

    records: List[Record] = []
    for root in roots:
        for path in _iter_files(root):
            records.append(_record_for_path(path, base_dir))

    records.sort(key=lambda r: r.path)

    out_path = Path(args.out).resolve() if args.out else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(records, out_path)

    if args.env_out:
        _write_env_snapshot(Path(args.env_out).resolve())

    return 0


def _diff(args: argparse.Namespace) -> int:
    baseline = _load_manifest(Path(args.baseline))
    current = _load_manifest(Path(args.current))

    baseline_paths = set(baseline.keys())
    current_paths = set(current.keys())

    removed = sorted(baseline_paths - current_paths)
    added = sorted(current_paths - baseline_paths)

    changed: List[Tuple[str, str]] = []
    for path in sorted(baseline_paths & current_paths):
        reason = _compare_records(baseline[path], current[path])
        if reason:
            changed.append((path, reason))

    has_diff = bool(removed or added or changed)

    if removed:
        print("Removed files:")
        for path in removed:
            print(f"- {path}")
    if added:
        print("Added files:")
        for path in added:
            print(f"- {path}")
    if changed:
        print("Changed files:")
        for path, reason in changed:
            print(f"- {path}: {reason}")

    if not has_diff:
        print("No differences detected.")

    return 1 if has_diff else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and diff artifact manifests for regression testing."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot", help="Create a manifest snapshot")
    snapshot.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="Artifact roots to scan",
    )
    snapshot.add_argument(
        "--out",
        required=False,
        help="Output TSV path (default: stdout)",
    )
    snapshot.add_argument(
        "--env_out",
        required=False,
        help="Environment snapshot output path",
    )
    snapshot.add_argument(
        "--base_dir",
        default=os.getcwd(),
        help="Base directory for relative paths (default: cwd)",
    )
    snapshot.set_defaults(func=_snapshot)

    diff = subparsers.add_parser("diff", help="Diff two manifests")
    diff.add_argument("--baseline", required=True, help="Baseline TSV path")
    diff.add_argument("--current", required=True, help="Current TSV path")
    diff.set_defaults(func=_diff)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
