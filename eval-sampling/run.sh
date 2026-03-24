#!/bin/bash

#SBATCH -c 12
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH -t 2-00:00
#SBATCH -p ${SLURM_PARTITION}
#SBATCH --mem=128GB
#SBATCH --account=barak_lab

module load cuda/12.4
module load gcc

source $SCRATCH/envs/control/bin/activate

# set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29533
export NCCL_SOCKET_FAMILY=AF_INET
export HF_HOME=${HF_HOME}

echo "Running evaluation for model $1"
task="leaderboard"
python eval.py --tasks $task --model_id $1 --batch_size 1 --output-dir results/${SLURM_JOB_NAME}

# get model name from a list of models
MODEL_NAME=$(cat data/top_models_by_base.csv | awk -F',' -v id=$1 'NR==id {print $2}')
MODEL_NAME=$(echo $MODEL_NAME | tr '/' '--')
rm -r $HF_HOME/huggingface/hub/models--$MODEL_NAME
echo "Cleared cache for model $MODEL_NAME at $HF_HOME/huggingface/hub/models--$MODEL_NAME"