MODEL_DIR=/remote-home1/zgliu/wrote_program/modelPruning/llm_shearing/ckpts/finetune/internlm2_7b_pruning_scaling_constant_to13b_sl4096_ft48000ba_20240902_014102
# MODEL_PATH=$MODEL_DIR/latest-rank0.pt
# python3 -m llmshearing.utils.post_pruning_processing prune_and_save_model $MODEL_PATH

MODEL_PATH=$MODEL_DIR/latest-rank0.pt
OUTPUT_PATH=$MODEL_DIR/hf-latest_rank0
MODEL_CLASS=LlamaForCausalLM
HIDDEN_SIZE=5120
NUM_ATTENTION_HEADS=40
NUM_HIDDEN_LAYERS=40
INTERMEDIATE_SIZE=13824
MODEL_NAME=Pruned-Moss-13B

python3 -m llmshearing.utils.composer_to_hf $MODEL_PATH $OUTPUT_PATH \
        model_class=${MODEL_CLASS} \
        hidden_size=${HIDDEN_SIZE} \
        num_attention_heads=${NUM_ATTENTION_HEADS} \
        num_hidden_layers=${NUM_HIDDEN_LAYERS} \
        intermediate_size=${INTERMEDIATE_SIZE} \
        num_key_value_heads=${NUM_ATTENTION_HEADS} \
        _name_or_path=${MODEL_NAME}