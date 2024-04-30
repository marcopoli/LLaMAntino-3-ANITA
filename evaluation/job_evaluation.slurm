#!/bin/bash
#SBATCH -A <PROJECT_NAME>
#SBATCH --job-name=llama3
#SBATCH --output=res_eval_anita.txt
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=04-00:00:00
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --gres=gpu:4
#SBATCH --mem=243000

module load profile/deeplrn
module load cuda/12.1
module load gcc/12.2.0-cuda-12.1
module load python/3.11.6--gcc--8.5.0
module load nccl/2.19.3-1--gcc--12.2.0-cuda-12.1
module load cudnn/8.9.7.29-12--gcc--12.2.0-cuda-12.1
module load cutensor/1.5.0.3--gcc--12.2.0-cuda-12.1
export HF_DATASETS_CACHE="/leonardo_scratch/large/userexternal/xxx/hf_datasets/datasets/"
unset TRANSFORMERS_CACHE
export HF_HOME="/leonardo_scratch/large/userexternal/xxx/hf_datasets/models/"
export PIP_CACHE_DIR="/leonardo_scratch/large/userexternal/xxx/cache/"

export LOGLEVEL=DEBUG


export NCCL_DEBUG=WARN;
export NCCL_DEBUG_SUBSYS=WARN
export NCCL_SOCKET_IFNAME=en,eth,em,bond,ens;
export CXX=g++;

export CUDA_LAUNCH_BLOCKING=1
export ACCELERATE_USE_FSDP=0

accelerate launch -m lm_eval --model hf --model_args pretrained=<MODEL> --tasks arc_challenge,hellaswag,mmlu,truthfulqa,winogrande,gsm8k --batch_size auto:4
