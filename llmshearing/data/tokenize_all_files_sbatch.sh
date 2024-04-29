#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --ntasks=1
#SBATCH -p moss
#SBATCH --mem=10240
#SBATCH --output=/remote-home/zyzeng/LLM-Shearing/LLM-Shearing/logs/tokenize/tokenize_%A_%a.out
#SBATCH --error=/remote-home/zyzeng/LLM-Shearing/LLM-Shearing/logs/tokenize/tokenize_%A_%a.err
#SBATCH --array=368-428
#SBATCH --nodelist=slurmd-9

srun python tokenize_single_file.py --target_dir /remote-home/zyzeng/LLM-Shearing/LLM-Shearing/data_bin/moss2.5b/ \
--raw_dir /remote-home/zyzeng/LLM-Shearing/LLM-Shearing/data_raw/moss_data_split/ \
--tokenizer /remote-home/share/models/moss2-2_5b-hf/tokenizer.model \
--text_key text

# srun python tokenize_single_file.py --target_dir /remote-home/zyzeng/LLM-Shearing/LLM-Shearing/data_bin/moss/ \
# --raw_dir /remote-home/zyzeng/LLM-Shearing/LLM-Shearing/data_raw/moss_data_split/redpajama/ \
# --tokenizer /remote-home/share/models/internlm2-7b-base2-hf/tokenizer.model 

# srun python tokenize_single_file.py --target_dir /remote-home/zyzeng/LLM-Shearing/LLM-Shearing/data_bin/moss/wanjuan/ \
# --raw_dir /remote-home/zyzeng/LLM-Shearing/LLM-Shearing/data_raw/moss_data_split/wanjuan/ \
# --tokenizer /remote-home/share/models/internlm2-7b-base2-hf/tokenizer.model \
# --text_key content