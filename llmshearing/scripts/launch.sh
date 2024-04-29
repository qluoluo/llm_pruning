#!/bin/bash
PROJ_DIR="/remote-home/zyzeng/LLM-Shearing/LLM-Shearing"

# num_nodes=$(scontrol show job $SLURM_JOB_ID | grep NodeList=della | wc -l)
num_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export MASTER_ADDR=$master_addr
echo $SLURM_GPUS_PER_NODE

export WORLD_SIZE=$(( $num_nodes * $SLURM_GPUS_PER_NODE ))
export MASTER_PORT=$(( 10000 + RANDOM % 10000 ))

echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE
echo "num_nodes="$num_nodes

if [[ $num_nodes == 1 ]]; then composer $PROJ_DIR/llmshearing/train.py $@; 
else srun bash $PROJ_DIR/llmshearing/scripts/srun_launch.sh $@; fi
 
