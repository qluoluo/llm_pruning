### 环境安装
```
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==1.0.3.post

cd llmshearing
pip install -r requirement.txt

pip install -e .
```

### 数据准备
处理新数据流程是preprocess -> tokenize -> sample -> merge eval data

- preprocess: 需要划分domain，每个domain需要均匀切分成多个文件 （不能只有一个文件），文件多一点用slurm并行处理更快
- tokenize: tokenize_all_files_sbatch.sh，运行这个之前需要运行llmshearing/data/get_all_jsonl.py获取文件列表，通过调整slurm的array来控制进程数量 & id
- sample: sample_all_domain_sbatch.sh，运行完会生成三个目录：for_prune, for_ft, eval 
- merge eval data: 运行merge_data.py，将eval下的数据merge到for_prune和for_ft

### 模型准备
需要将模型权重转换为 Composer 预期的格式
```
# Define the Hugging Face model name and the output path
HF_MODEL_NAME=hf/model/path
OUTPUT_PATH=output/state_dict.pt

# Create the necessary directory if it doesn't exist
mkdir -p $(dirname $OUTPUT_PATH)

# Convert the Hugging Face model to Composer key format
python3 -m llmshearing.utils.hf_to_composer $HF_MODEL_NAME $OUTPUT_PATH
```

### 剪枝代码

```
llmshearing/scripts/pruning_20b_to_13b.sh
```

### 续训代码

```
llmshearing/scripts/continue_pretrain_13b.sh
```