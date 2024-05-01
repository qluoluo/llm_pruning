# 2.5b Model剪枝
## 数据
最麻烦的地方是数据处理，不过数据已经处理好了，在/remote-home/share/personal/zyzeng/data/moss2.5b_sampled/

如果要自己处理新数据，流程是preprocess -> tokenize -> sample -> merge eval data

- preprocess: 需要划分domain，每个domain需要均匀切分成多个文件 （不能只有一个文件），文件多一点用slurm并行处理更快
- tokenize: tokenize_all_files_sbatch.sh，运行这个之前需要运行llmshearing/data/get_all_jsonl.py获取文件列表，通过调整slurm的array来控制进程数量 & id
- sample: sample_all_domain_sbatch.sh，运行完会生成三个目录：for_prune, for_ft, eval 
- merge eval data: 运行merge_data.py，将eval下的数据merge到for_prune和for_ft

## 训练
已经转换为composer格式的2.5b路径：/remote-home/share/personal/zyzeng/models/moss2_2.5b_composer.pt

- 模型的config文件: llmshearing/configs/internlm/2.5b.yaml
- 剪枝的shell文件：llmshearing/scripts/pruning_2.5b_to_100m.sh
- 续训shell文件：参考llmshearing/scripts/continue_pretrain.sh

需要调整一下路径，这个项目里面不少shell文件都写入了绝对路径，可以通过全局搜索zyzeng找到这些地方替换掉

prune的日志会被写入到logs/prune下面

## 问题
在llmshearing/scripts/pruning_2.5b_to_100m.sh中，加了一个自定义的config：frozen_embedding。在train.py可以通过cfg.frozen_embedding读取这个参数。

我们想要frozen embedding的参数，但是设置requires_grad=False不行，和FSDP是冲突的。当前改的是build_optimizer，创建了一个新的group，将embedding的lr设置为0。但是其它需要训练的参数的lr也变成了0，而且一直不变，这是scheduler出了问题，我猜。
