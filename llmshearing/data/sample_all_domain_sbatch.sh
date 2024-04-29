#!/bin/bash
#SBATCH --job-name=sample
#SBATCH --ntasks=1
#SBATCH -p moss
#SBATCH --cpus-per-task=1
#SBATCH --output=/remote-home/zyzeng/LLM-Shearing/LLM-Shearing/logs/sample/sample_%A_%a.out
#SBATCH --error=/remote-home/zyzeng/LLM-Shearing/LLM-Shearing/logs/sample/sample_%A_%a.err
#SBATCH --array=7

srun python sample_moss_data.py --target_dir ../../data_bin/moss2.5b_sampled/ --tokenized_dir ../../data_bin/moss2.5b/ --for_prune 1.0 --for_ft 100 --eval_seq 50
