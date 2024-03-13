#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --ntasks=1
#SBATCH -p moss
#SBATCH --cpus-per-task=1
#SBATCH --output=/remote-home/zyzeng/LLM-Shearing/LLM-Shearing/logs/tokenize/tokenize_%A_%a.out
#SBATCH --error=/remote-home/zyzeng/LLM-Shearing/LLM-Shearing/logs/tokenize/tokenize_%A_%a.err
#SBATCH --array=0-503

srun python tokenize_single_file.py --target_dir /remote-home/zyzeng/LLM-Shearing/LLM-Shearing/data_bin/moss/ \
--raw_dir /remote-home/zyzeng/LLM-Shearing/LLM-Shearing/data_raw/moss_data_split/ \
--tokenizer /remote-home/share/models/internlm2-7b-base2-hf/tokenizer.model 