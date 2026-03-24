#!/usr/bin/env python3
"""
Model evaluation script using lm-eval for models from CSV data.

This script reads model information from the CSV file and runs evaluations
using the lm-eval framework.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import shutil
import glob
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Handles model evaluation using lm-eval."""
    
    def __init__(self, csv_path: str, output_dir: str = "results/lm-eval", batch_size: int = -1):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.models_df = self.load_models()
        self.batch_size = batch_size
        
    def load_models(self) -> pd.DataFrame:
        """Load models from CSV file."""
        try:
            df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(df)} models from {self.csv_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            raise
    
    def pre_download_datasets(self, tasks: List[str] = None) -> bool:
        """
        Pre-download datasets to avoid rate limiting when multiple processes try to download simultaneously.
        Returns True if successful, False otherwise.
        """
        if tasks is None:
            tasks = ["leaderboard"]
        
        logger.info(f"Pre-downloading datasets for tasks: {tasks}")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="DigitalLearningGmbH/MATH-lighteval",
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
            local_dir=os.path.join(os.environ["HF_HUB_CACHE"], "snapshots", "MATH-lighteval"),
            local_dir_use_symlinks=False
        )
        
        try:
            # Import datasets library
            from datasets import load_dataset
            from huggingface_hub import login
            
            # Login if token is available
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                try:
                    login(token=hf_token, add_to_git_credential=False)
                except Exception as e:
                    logger.warning(f"Failed to login to HuggingFace: {e}")
            
            # For leaderboard task, we need to download the BBH dataset
            # The leaderboard task typically includes multiple subtasks that use BBH
            if "leaderboard" in tasks:
                logger.info("Pre-downloading BBH dataset (used by leaderboard tasks)...")
                try:
                    # Try to load the dataset that's causing the issue
                    # This will cache it locally so all processes can use the cache
                    dataset = load_dataset(
                        "SaylorTwift/bbh",
                        trust_remote_code=True,
                        token=hf_token
                    )
                    # Handle both DatasetDict and Dataset objects
                    if hasattr(dataset, 'keys'):
                        logger.info(f"Successfully pre-downloaded BBH dataset: {len(dataset)} splits")
                    else:
                        logger.info("Successfully pre-downloaded BBH dataset")
                except Exception as e:
                    logger.warning(f"Failed to pre-download BBH dataset: {e}")
                    logger.info("Will attempt to download during evaluation (may hit rate limits)")
            
            # Add a small delay to avoid immediate rate limiting
            time.sleep(2)
            
            logger.info("Dataset pre-download complete")
            return True
            
        except ImportError:
            logger.warning("datasets library not available, skipping pre-download")
            return False
        except Exception as e:
            logger.warning(f"Error pre-downloading datasets: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def is_peft_adapter(self, model_name: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a model is a PEFT adapter.
        Returns: (is_peft, base_model_name_or_path)
        """
        try:
            from peft import PeftConfig
            
            logger.info(f"Checking if {model_name} is a PEFT adapter...")
            
            # Try to load PEFT config directly - this is the most reliable method
            try:
                hf_token = os.environ.get("HF_TOKEN")
                peft_config = PeftConfig.from_pretrained(model_name, token=hf_token)
                base_model = peft_config.base_model_name_or_path
                
                if base_model:
                    logger.info(f"✅ Detected PEFT adapter: {model_name} -> base model: {base_model}")
                    return True, base_model
                else:
                    logger.warning(f"PEFT config found but no base_model_name_or_path: {model_name}")
                    return False, None
            except Exception as e:
                # If PeftConfig.from_pretrained fails, it's not a PEFT adapter
                logger.debug(f"Not a PEFT adapter (PeftConfig.from_pretrained failed): {e}")
                return False, None
        except ImportError:
            logger.warning("PEFT library not available, cannot check for PEFT adapters")
            return False, None
        except Exception as e:
            logger.warning(f"Error checking for PEFT adapter: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False, None
    
    def run_evaluation(self, 
                      model_id: int,
                      model_name: str,
                      base_model_name: str,
                      tasks: List[str] = None,
                      limit: Optional[int] = None,
                      output_file: Optional[str] = None,
                      timeout: int = 259200) -> Dict:
        """Run evaluation for a single model."""
        if tasks is None:
            tasks = ["leaderboard"]
        
        if output_file is None:
            output_file = f"{self.output_dir}/results.json"

        env = os.environ.copy()
        # Get master port and address - use consistent values
        master_port = os.environ.get("MASTER_PORT", "29533")
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        
        env.update({
            "ACCELERATE_DISABLE_RICH": "1",
            "TRANSFORMERS_VERBOSITY": "error",
            "HF_DATASETS_VERBOSITY": "error",
            "TOKENIZERS_PARALLELISM": "false",
            # Prefer offline/cached datasets to avoid rate limiting
            "HF_HUB_OFFLINE": "0",  # Set to "1" to force offline mode
            "HF_DATASETS_OFFLINE": "0",  # Set to "1" to force offline mode
            # Add retry configuration for HuggingFace Hub
            "HF_HUB_DOWNLOAD_TIMEOUT": "15000",  # 50 minutes timeout
            # Use local cache aggressively
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            # Enable better error reporting for distributed training
            # Each rank will write to its own error file
            "TORCHELASTIC_ERROR_FILE": os.path.join(self.output_dir, "torchelastic_error.json"),
            "TORCHELASTIC_MAX_RESTARTS": "0",  # Don't restart on failure, just fail fast
            # "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",  # Help with CUDA memory fragmentation
            # Enable Python traceback for better error visibility
            "PYTHONUNBUFFERED": "1",  # Unbuffered output for better error visibility
            # Distributed training environment variables (fix socket connection errors)
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": master_port,
            # Force IPv4 for socket connections (fixes errno 97 - Address family not supported)
            "NCCL_SOCKET_FAMILY": "AF_INET",  # Force IPv4
            "NCCL_IB_DISABLE": "1",  # Disable InfiniBand if not available
            "NCCL_P2P_DISABLE": os.environ.get("NCCL_P2P_DISABLE", "0"),  # Enable P2P if available
            # Additional debugging for distributed training (set to WARN to reduce noise)
            "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
            # Force IPv4 for PyTorch distributed
            "GLOO_SOCKET_IFNAME": "lo",  # Use loopback interface
            "TP_SOCKET_IFNAME": "lo",  # Use loopback interface for tensor parallel
        })
        
        # Add HF_TOKEN to environment if available
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            env["HF_TOKEN"] = hf_token
            env["HUGGING_FACE_HUB_TOKEN"] = hf_token
        
        # Check if this is a PEFT adapter and handle it
        is_peft, base_model = self.is_peft_adapter(model_name)
        
        # Build lm-eval command
        if is_peft:
            model_args = f"pretrained={base_model},trust_remote_code=true,dtype=auto"
            model_args += f",peft={model_name}"
        else:
            model_args = f"pretrained={model_name},trust_remote_code=true,dtype=auto"
        
        if "awq" in model_name.lower():
            model_args += f",awq=True"
    
        if hf_token:
            model_args += f",token={hf_token}"
        logger.info(f"Using model path for lm-eval: {model_name}")
        
        # Determine number of processes (GPUs)
        # Check CUDA_VISIBLE_DEVICES if set, otherwise let accelerate auto-detect
        num_processes = None
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            if cuda_devices:
                # Count number of devices (comma-separated)
                num_processes = str(cuda_devices.count(",") + 1)
        
        logger.info(f"Number of processes: {num_processes}")
        
        # Build accelerate launch command with explicit port configuration
        # This ensures accelerate uses the same port as MASTER_PORT environment variable
        cmd = [
            "accelerate", "launch",
            "--main_process_port", master_port,  # Match MASTER_PORT
        ]
        
        # Add num_processes only if we can determine it, otherwise let accelerate auto-detect
        if num_processes:
            cmd.extend(["--num_processes", num_processes])
        # For single GPU, use device_map=auto for automatic device placement
        # For multi-GPU with accelerate, do NOT use device_map=auto as accelerate handles device placement
        # and device_map=auto can cause device mismatches (some weights on CPU, inputs on CUDA)
        if num_processes and int(num_processes) == 1:
            model_args += f",device_map=auto"
        
        # Add the rest of the command
        cmd.extend([
            "-m", "utils.quiet_wrap",
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", model_args,
            "--tasks", ",".join(tasks),
            "--gen_kwargs", "do_sample=True,top_p=0.8,temperature=0.2", # "do_sample=True,top_p=0.95,temperature=0.9",
            "--batch_size", "auto" if self.batch_size == -1 else str(self.batch_size),
            "--output_path", output_file,
            "--log_samples"
        ])
        
        if limit:
            cmd.extend(["--limit", str(limit)])
        
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            # Use Popen for better control and real-time output
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                env=env
            )
            
            output_lines = []
            import time
            start_time = time.time()
            timeout_seconds = timeout if timeout > 0 else None  # 0 or negative means no timeout
            last_progress_log = start_time
            progress_interval = 300  # Log progress every 5 minutes
            
            while True:
                elapsed = time.time() - start_time
                
                # Check for timeout (only if timeout is set)
                if timeout_seconds is not None and elapsed > timeout_seconds:
                    logger.warning(f"Evaluation timeout reached ({timeout_seconds}s) for {model_id}, terminating process...")
                    process.terminate()
                    
                    # Give process a grace period to clean up
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Process did not terminate gracefully, forcing kill...")
                        process.kill()
                        process.wait()
                    
                    logger.error(f"Evaluation timed out for {model_id} after {int(elapsed)}s (timeout: {timeout_seconds}s)")
                    return {"status": "timeout", "error": f"Evaluation timed out after {int(elapsed)} seconds"}
                
                # Log progress periodically
                if time.time() - last_progress_log > progress_interval:
                    elapsed_min = int(elapsed / 60)
                    if timeout_seconds:
                        remaining_min = int((timeout_seconds - elapsed) / 60)
                        logger.info(f"[{model_id}] Evaluation in progress: {elapsed_min}min elapsed, {remaining_min}min remaining")
                    else:
                        logger.info(f"[{model_id}] Evaluation in progress: {elapsed_min}min elapsed (no timeout)")
                    last_progress_log = time.time()
                
                # Check if process is still running
                if process.poll() is not None:
                    # Process finished, read any remaining output
                    remaining_output = process.stdout.read()
                    if remaining_output:
                        for line in remaining_output.strip().split('\n'):
                            if line.strip():
                                output_lines.append(line.strip())
                                logger.info(f"[{model_id}] {line.strip()}")
                    break
                
                # Try to read a line with a short timeout
                try:
                    import select
                    if select.select([process.stdout], [], [], 1.0)[0]:  # 1 second timeout
                        output = process.stdout.readline()
                        if output:
                            output_lines.append(output.strip())
                            logger.info(f"[{model_id}] {output.strip()}")
                except (ImportError, OSError):
                    # Fallback for systems without select module or non-Unix systems
                    output = process.stdout.readline()
                    if output:
                        output_lines.append(output.strip())
                        logger.info(f"[{model_id}] {output.strip()}")
                    time.sleep(0.1)  # Small delay to prevent busy waiting
            
            returncode = process.poll()
            
            if returncode == 0:
                logger.info(f"Evaluation completed for {model_id}")
                return {"status": "success", "output": "\n".join(output_lines)}
            else:
                logger.error(f"Evaluation failed for {model_id} with return code {returncode}")
                
                # Analyze error output for common issues
                error_output = "\n".join(output_lines)
                error_summary = []
                
                # Check for socket errors
                if "socket" in error_output.lower() or "errno: 97" in error_output:
                    error_summary.append("Socket connection error detected. Check MASTER_ADDR, MASTER_PORT, and NCCL_SOCKET_FAMILY environment variables.")
                    error_summary.append("See DEBUGGING.md for solutions.")
                
                # Check for distributed training errors
                if "ChildFailedError" in error_output or "torch.distributed" in error_output:
                    error_summary.append("Distributed training error detected.")
                    error_summary.append("Check network configuration and ensure all GPUs can communicate.")
                
                # Check for timeout
                if returncode == 245 or "SIGTERM" in error_output:
                    error_summary.append("Process was terminated (SIGTERM).")
                    error_summary.append("Possible causes: timeout, resource limits, or parent process termination.")
                
                # Check for CUDA errors
                if "cuda" in error_output.lower() and "error" in error_output.lower():
                    error_summary.append("CUDA error detected. Check GPU availability and memory.")
                
                if error_summary:
                    logger.error("Error analysis:\n" + "\n".join(f"  - {msg}" for msg in error_summary))
                
                return {
                    "status": "error", 
                    "error": error_output,
                    "returncode": returncode,
                    "error_summary": error_summary
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"Evaluation timed out for {model_id}")
            return {"status": "timeout", "error": "Evaluation timed out"}
        except Exception as e:
            logger.error(f"Unexpected error evaluating {model_id}: {e}")
            return {"status": "error", "error": str(e)}
    
    def evaluate_models(self, 
                       model_id: int,
                       model_name: str,
                       base_model_name: str,
                       tasks: List[str] = None,
                       limit: Optional[int] = None,
                       timeout: int = 259200,
                       pre_download: bool = True) -> Dict:
        """Evaluate a single model."""
        
        logger.info(f"Evaluating model {model_id} -- {model_name}")
        
        # Pre-download datasets to avoid rate limiting with multiple processes
        if pre_download:
            logger.info("Pre-downloading datasets to avoid rate limiting...")
            self.pre_download_datasets(tasks=tasks)
            # Add a delay after pre-download to let rate limit reset
            time.sleep(5)
            
        result = self.run_evaluation(
                model_id=model_id,
                model_name=model_name,
                base_model_name=base_model_name,
                tasks=tasks,
                limit=limit,
                timeout=timeout
            )
        
        # Clear HuggingFace cache after inference
        logger.info(f"Clearing HuggingFace cache for {model_name}")
        self.clear_model_cache(model_name)
            
        return result
    
    def clear_model_cache(self, model_name: str) -> bool:
        """Clear HuggingFace cache for a specific model after successful inference."""
        try:
            # Get HuggingFace cache directories (check multiple possible locations)
            cache_dirs = []
            
            # Check HF_HOME environment variable
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                hf_hub = Path(hf_home) / "huggingface" / "hub"
                cache_dirs.append(hf_hub)
            
            # Check TRANSFORMERS_CACHE environment variable
            transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
            if transformers_cache:
                cache_dirs.append(Path(transformers_cache))
            
            # Default HuggingFace cache location
            default_cache = Path.home() / ".cache" / "huggingface" / "hub"
            if default_cache not in cache_dirs:
                cache_dirs.append(default_cache)
            
            # Model name format: "org/model" -> HuggingFace uses "models--org--model-name"
            model_cache_name = f"models--{model_name.replace('/', '--')}"
            found_any = False

            if hf_home:
                cache_model_dir = hf_hub / model_cache_name
                if cache_model_dir.exists():
                    logger.info(f"Directly clearing cache for model {model_name} at {cache_model_dir}")
                    try:
                        shutil.rmtree(cache_model_dir, ignore_errors=True)
                        logger.info(f"Successfully cleared cache for model {model_name} at {cache_model_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to remove cache directory {cache_model_dir}: {e}")
            
            for cache_dir in cache_dirs:
                if not cache_dir.exists():
                    continue
                
                # Try to find and remove model cache directory
                cache_model_dir = cache_dir / model_cache_name
                
                if cache_model_dir.exists():
                    logger.info(f"Clearing cache for model {model_name} at {cache_model_dir}")
                    try:
                        # Calculate size before deletion for logging
                        total_size = sum(f.stat().st_size for f in cache_model_dir.rglob('*') if f.is_file())
                        size_mb = total_size / (1024 * 1024)
                        
                        shutil.rmtree(cache_model_dir)
                        logger.info(f"Successfully cleared {size_mb:.2f} MB of cache for {model_name}")
                        found_any = True
                    except Exception as e:
                        logger.warning(f"Failed to remove cache directory {cache_model_dir}: {e}")
                else:
                    # Also check for any subdirectories that might match (fallback)
                    try:
                        for item in cache_dir.iterdir():
                            if item.is_dir() and model_name.replace("/", "--") in item.name:
                                logger.info(f"Found matching cache directory: {item}, removing...")
                                try:
                                    total_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                                    size_mb = total_size / (1024 * 1024)
                                    shutil.rmtree(item)
                                    logger.info(f"Successfully cleared {size_mb:.2f} MB from cache directory: {item}")
                                    found_any = True
                                except Exception as e:
                                    logger.warning(f"Failed to remove cache directory {item}: {e}")
                    except (PermissionError, OSError) as e:
                        # Skip if we can't read the directory
                        logger.debug(f"Could not read cache directory {cache_dir}: {e}")
            
            if not found_any:
                logger.info(f"No cache found for model {model_name} in any cache directory")
            
            return found_any
                
        except Exception as e:
            logger.warning(f"Error clearing cache for {model_name}: {e}")
            return False
    
    def generate_summary_report(self, result: Dict) -> str:
        """Generate a summary report of evaluation results."""
        status = result.get("status", "unknown")
        return f"Evaluation Status: {status}"

def main():
    parser = argparse.ArgumentParser(description="Evaluate models using lm-eval")
    
    # Job parameters
    parser.add_argument("--batch_size", type=int, default=-1,
                       help="Batch size for evaluation (-1 for auto)")
    # Input/Output
    parser.add_argument("--csv", default="data/top_models_by_base.csv",
                       help="Path to CSV file with model data")
    # Model selection
    parser.add_argument("--model_id", type=int, required=True,
                       help="A single model id to evaluate")
    parser.add_argument("--output-dir", default="results/lm-eval",
                       help="Output directory for results")
    
    # Evaluation parameters
    parser.add_argument("--tasks", nargs="+", 
                       default=["leaderboard"],
                       help="Evaluation tasks to run")
    parser.add_argument("--eval-limit", type=int,
                       help="Limit number of examples per task")
    parser.add_argument("--timeout", type=int, default=0,
                       help="Timeout in seconds for each evaluation (0 to disable)")
    parser.add_argument("--pre-download", action="store_true", default=True,
                       help="Pre-download datasets before evaluation to avoid rate limiting")
    parser.add_argument("--no-pre-download", dest="pre_download", action="store_false",
                       help="Skip pre-downloading datasets (may hit rate limits)")
    
    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ModelEvaluator(args.csv, args.output_dir, args.batch_size)
    
    # Load model information
    logger.info(f"Using specified model id: {args.model_id}")
    model_name = evaluator.models_df.loc[args.model_id, 'model_id']
    base_model_name = evaluator.models_df.loc[args.model_id, 'mapped_base_model']
    
    if "mlx" in base_model_name.lower():
        with open(f"{args.output_dir}/summary_report.txt", "w") as f:
            f.write("Skip MLX models")
        sys.exit(0)
    
    # Set up output directory
    scratch_dir = os.environ.get("SCRATCH", "/n/netscratch/kempner_sham_lab/Lab/hanlinzhang")
    args.output_dir = f"{scratch_dir}/control-scale/{args.output_dir}/{args.model_id}/{model_name.replace('/', '_')}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    evaluator.output_dir = args.output_dir

    # Check if results_*.json file already exists
    results_pattern = os.path.join(args.output_dir, "results_*.json")
    existing_results = glob.glob(results_pattern)
    if existing_results:
        logger.info(f"Found existing results file(s): {existing_results}")
        logger.info(f"Skipping evaluation for model {args.model_id} ({model_name}) to avoid redoing the eval")
        sys.exit(0)

    logger.info(f"Running evaluation for the {args.model_id}th model -- {model_name}")
    
    # Run evaluation
    result = evaluator.evaluate_models(
        model_id=args.model_id,
        base_model_name=base_model_name,
        model_name=model_name,
        tasks=args.tasks,
        limit=args.eval_limit,
        timeout=args.timeout,
        pre_download=args.pre_download
    )
    
    # Generate summary report
    summary = evaluator.generate_summary_report(result)
    logger.info(summary)
    
    # Save final results
    with open(f"{args.output_dir}/final_results.json", "w") as f:
        json.dump(result, f, indent=2)
    
    with open(f"{args.output_dir}/summary_report.txt", "w") as f:
        f.write(summary)
    
    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
