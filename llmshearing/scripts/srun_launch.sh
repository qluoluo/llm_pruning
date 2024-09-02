PROJ_DIR=/remote-home1/zgliu/wrote_program/modelPruning/llm_shearing
cd ${PROJ_DIR}
echo ${SLURM_NODEID} 
composer --node_rank ${SLURM_NODEID} $PROJ_DIR/llmshearing/train.py $@  