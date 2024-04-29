PROJ_DIR=/remote-home/zyzeng/LLM-Shearing/LLM-Shearing
cd ${PROJ_DIR}
echo ${SLURM_NODEID} 
composer --node_rank ${SLURM_NODEID} $PROJ_DIR/llmshearing/train.py $@  