PROJ_DIR=/remote-home/zgliu/wrote_program/modelPruning/llm_shearing
cd ${PROJ_DIR}
echo ${SLURM_NODEID} 
composer --node_rank ${SLURM_NODEID} $PROJ_DIR/llmshearing/train.py $@  