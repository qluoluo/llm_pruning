#!/bin/bash
#SBATCH --job-name=run_eval
#SBATCH --ntasks=1
#SBATCH -p moss
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/remote-home/zyzeng/LLM-Shearing/LLM-Shearing/logs/run_eval/run_eval_%A_%a.out
#SBATCH --error=/remote-home/zyzeng/LLM-Shearing/LLM-Shearing/logs/run_eval/run_eval_%A_%a.err

# composer_tokenizer_path = "/remote-home/zyzeng/LLM-Shearing/LLM-Shearing/ckpts/mha_internlm2_hf/tokenizer.model"
# default model dirs
# hf_model_path = "/remote-home/zyzeng/LLM-Shearing/LLM-Shearing/ckpts/mha_internlm2_hf/"
# composer_model_path = "/remote-home/zyzeng/LLM-Shearing/LLM-Shearing/ckpts/internlm2_composer.pt"
python_script="/remote-home/zyzeng/LLM-Shearing/LLM-Shearing/scaling_law/caculate_loss.py"
python_bin="/remote-home/zyzeng/miniconda3/envs/prune/bin/python"
eval composer 1.3b
srun python ${python_script} --model_type composer \
    --model_path /remote-home/zyzeng/LLM-Shearing/LLM-Shearing/ckpts/internlm_7b_pruning_scaling_constant_to1.3b_sl4096_v2/pruned-latest-rank0.pt \
    --tokenizer_path /remote-home/zyzeng/LLM-Shearing/LLM-Shearing/ckpts/mha_internlm2_hf/tokenizer.model \
    --cfg_path llmshearing/configs/internlm/1.3b.yaml

# eval internlm-composer-7b
# srun python ${python_script} --model_type composer \
#     --model_path ckpts/internlm2_composer.pt \
#     --tokenizer_path /remote-home/zyzeng/LLM-Shearing/LLM-Shearing/ckpts/mha_internlm2_hf/tokenizer.model \
#     --cfg_path llmshearing/configs/internlm/7b.yaml

# eval internlm-1.8b
# srun python ${python_script} --model_type hf \
#     --model_path /remote-home/share/models/internlm2-1_8b/ \
#     --tokenizer_path /remote-home/share/models/internlm2-1_8b/

# eval honor2_5b
# srun ${python_bin} ${python_script} --model_type hf \
#     --model_path /remote-home/share/models/moss2-2.5b-enhance-hf/official_Tangwan_2.5B_1.1.0_Enhance_1.3.0/50000 \
#     --tokenizer_path /remote-home/share/models/moss2-2.5b-enhance-hf/official_Tangwan_2.5B_1.1.0_Enhance_1.3.0/50000

# eval honor_2.5b_pro
# srun ${python_bin} ${python_script} --model_type hf \
#     --model_path /remote-home/share/models/moss2-2.5b-enhance-hf/official_Tangwan_2.5B_1.1.0_Enhance_1.3.0/50000 \
#     --tokenizer_path /remote-home/share/models/moss2-2.5b-enhance-hf/official_Tangwan_2.5B_1.1.0_Enhance_1.3.0/50000

# eval honor_2.5b_pro
# srun ${python_bin} ${python_script} --model_type hf \
#     --model_path /remote-home/share/models/moss2-2.5b-enhance-hf/official_Tangwan_2.5B_1.1.0_Enhance_1.3.0/50000 \
#     --tokenizer_path /remote-home/share/models/moss2-2.5b-enhance-hf/official_Tangwan_2.5B_1.1.0_Enhance_1.3.0/50000

