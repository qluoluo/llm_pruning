PROJ_DIR=/remote-home/rypeng/0224/LLM-Shearing

echo ${SLURM_NODEID} 
composer --node_rank ${SLURM_NODEID} $PROJ_DIR/llmshearing/train.py $@  