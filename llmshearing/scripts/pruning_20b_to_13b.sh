# pruning llama2 7b -> 3b or 1.3b

# Please specify the working folder
# PROJ_DIR=/scratch/gpfs/mengzhou/space2/LLM-Shearing
# LAUNCH_SCRIPT=${PROJ_DIR}/llmshearing/scripts/launch.sh
# DATA_DIR=/scratch/gpfs/mengzhou/llm_data/version5-uint32/500b_dedup_4k/for_prune
# OUTPUT_DIR=/scratch/gpfs/mengzhou/space2/out/test_release_pruning_full
# TRAIN_SCRIPT=${PROJ_DIR}/llmshearing/train.py
# MODEL_PATH=/projects/DANQIC/mengzhou/LLaMA2

PROJ_DIR=/remote-home1/zgliu/wrote_program/modelPruning/llm_shearing
LAUNCH_SCRIPT=${PROJ_DIR}/llmshearing/scripts/launch.sh
# DATA_DIR=/remote-home1/zgliu/wrote_program/modelPruning/llm_shearing/llmshearing/data/sample_redpajama
DATA_DIR=${PROJ_DIR}/data_bin/moss2.5b_sampled/for_prune
OUTPUT_DIR=${PROJ_DIR}/ckpts/
TRAIN_SCRIPT=${PROJ_DIR}/llmshearing/train.py
MODEL_PATH=${PROJ_DIR}/ckpts/
current_time=$(date +%Y%m%d_%H%M%S)
# Specify $PROJ_DIR in scripts/launch.sh and scripts/srun_launch.sh if using slurm

test=False

from_model=20b # source model size
to_model=13b # target model size
config_file=${PROJ_DIR}/llmshearing/configs/internlm/${from_model}.yaml

# path=$MODEL_PATH/moss2-20b-hf-bin-llama-mha-composer.pt
path=$MODEL_PATH/moss2-20b-hf-bin-llama-mha-composer-bf16-true.pt

# data setup
data_local=${DATA_DIR}

# basic setup
use_gpu_num=7
use_cpu_num=$((use_gpu_num * 16))
max_seq_len=4096
device_train_microbatch_size=2
global_train_batch_size=$((use_gpu_num * 2 * 8))
device_eval_batch_size=4

# learning setup
lr=1e-4 # learning rate for the main parameters
max_duration=3200ba # 0.42B tokens
save_interval=800ba # save in the end

save_weights_only=True

t_warmup=320ba # 10% learning rate warmup 

# dynamic loading setup
dynamic=True
set_names=[arxiv,cn_baike,cn_book,cn_weixin,cn_zhihu,code_starcoder,en_book,en_stackexchange,wanjuan,wiki] # domain names
proportion=[0.0038885832078429925,0.013171063389765087,6.913036813943098e-06,0.11492975550956504,0.06456880079775063,0.16964341743831857,0.0005333269641220821,0.07731814128971792,0.4785982974469239,0.07734170091917984] # initial proportion of RP, make sure that the sum(proportion) = 1
# doremi: update weights with exponential descent
# constant: keep the weights constant
update_type=constant 
if [[ $to_model == 1.3b ]]; then
    target_loss=[1.2465803623199463,3.0522592067718506,2.5978455543518066,2.5219357013702393,3.025589942932129,1.0684151649475098,2.2368016242980957,1.7007801532745361,2.288639783859253,1.634076714515686] # 1.3b predicted loss from scaling law
elif [[ $to_model == 2.7b ]]; then
    target_loss=[1.2465803623199463,3.0522592067718506,2.5978455543518066,2.5219357013702393,3.025589942932129,1.0684151649475098,2.2368016242980957,1.7007801532745361,2.288639783859253,1.634076714515686] # 2.7b predicted loss from scaling law
elif [[ $to_model == 370m ]]; then
    target_loss=[1.2465803623199463,3.0522592067718506,2.5978455543518066,2.5219357013702393,3.025589942932129,1.0684151649475098,2.2368016242980957,1.7007801532745361,2.288639783859253,1.634076714515686] # 410m predicted loss from scaling law
elif [[ $to_model == 100m ]]; then
    target_loss=[1.2465803623199463,3.0522592067718506,2.5978455543518066,2.5219357013702393,3.025589942932129,1.0684151649475098,2.2368016242980957,1.7007801532745361,2.288639783859253,1.634076714515686] # 410m predicted loss from scaling law
elif [[ $to_model == 13b ]]; then
    target_loss=[1.2465803623199463,3.0522592067718506,2.5978455543518066,2.5219357013702393,3.025589942932129,1.0684151649475098,2.2368016242980957,1.7007801532745361,2.288639783859253,1.634076714515686] # 410m predicted loss from scaling law
fi
eval_split_name=eval_merge # eval on all domains
eval_target_model=false # evaluate on the current model, not the target model, otherwise the loss will be inaccurate
eval_interval=50ba # eval every 50 batches and update the loading proportion


# pruning setup
lag_lr=1.0 # learning rate or l0_module
lagr_warmup=640ba # 20% sparsity warmup
if [[ $to_model == 1.3b ]]; then
    target_d_model=2048; target_n_heads=16; target_n_layers=24; target_intermediate_size=5504; target_vocab_size=92544
elif [[ $to_model == 2.7b ]]; then
    target_d_model=2560; target_n_heads=20; target_n_layers=32; target_intermediate_size=6912; target_vocab_size=92544
elif [[ $to_model == 370m ]]; then
    target_d_model=1024; target_n_heads=8; target_n_layers=24; target_intermediate_size=2816; target_vocab_size=92544
elif [[ $to_model == 100m ]]; then
    target_d_model=1024; target_n_heads=16; target_n_layers=5; target_intermediate_size=4096; target_vocab_size=137728
elif [[ $to_model == 13b ]]; then
    target_d_model=5120; target_n_heads=40; target_n_layers=40; target_intermediate_size=13824; target_vocab_size=137728
fi
# save directroy
run_name=moss_${from_model}_pruning_scaling_${update_type}_to${to_model}_sl${max_seq_len}_${current_time}
save_dir=${OUTPUT_DIR}/${run_name}
tensorboard_dir=${save_dir} # save locally

# Run in bash, it will automatically use resources available in the current environment
# composer $TRAIN_SCRIPT \
frozen_embedding=False
# Run with slurm
#    --mem=950gb \
#    --nodelist=slurmd-5,slurmd-9

# echo "before sbatch"

sbatch --job-name ${run_name} \
    --partition=huawei \
    --nodes=2 \
    --mem=950gb \
    --gpus-per-node=${use_gpu_num} \
    --cpus-per-task=${use_cpu_num} \
    --output=/remote-home1/zgliu/wrote_program/modelPruning/llm_shearing/logs/prune_time/prune_${current_time}_%A_%a.out \
    --error=/remote-home1/zgliu/wrote_program/modelPruning/llm_shearing/logs/prune_time/prune_${current_time}_%A_%a.err \
    $LAUNCH_SCRIPT \
    $config_file \
    run_name=${run_name} \
    data_local=${data_local} \
    eval_loader.dataset.split=${eval_split_name} \
    global_train_batch_size=${global_train_batch_size} \
    device_train_microbatch_size=${device_train_microbatch_size} \
    device_eval_batch_size=${device_eval_batch_size} \
    max_seq_len=${max_seq_len} \
    max_duration=${max_duration} \
    eval_first=true \
    scheduler.t_warmup=${t_warmup} \
    save_folder=${save_dir} \
    eval_interval=${eval_interval} \
    save_interval=${save_interval} \
    optimizer.lr=${lr} \
    optimizer.lag_lr=${lag_lr} \
    model.path=${path} \
    model.l0_module.lagrangian_warmup_steps=${lagr_warmup} \
    model.l0_module.pruning_modules='[head,intermediate,layer,hidden]' \
    model.l0_module.eval_target_model=${eval_target_model} \
    model.l0_module.target_model.d_model=${target_d_model} \
    model.l0_module.target_model.n_heads=${target_n_heads} \
    model.l0_module.target_model.n_layers=${target_n_layers} \
    model.l0_module.target_model.vocab_size=${target_vocab_size} \
    model.l0_module.target_model.intermediate_size=${target_intermediate_size} \
    model.l0_module.start_sparsity=0 \
    callbacks.data_loading.dynamic=${dynamic} \
    callbacks.data_loading.set_names=${set_names} \
    callbacks.data_loading.proportion=${proportion} \
    callbacks.data_loading.update_type=${update_type} \
    callbacks.data_loading.target_loss=${target_loss} \
    train_loader.num_workers=0 \
    train_loader.prefetch_factor=null \
    train_loader.persistent_workers=false \
    autoresume=false \
    frozen_embedding=${frozen_embedding} \
    save_weights_only=${save_weights_only} \