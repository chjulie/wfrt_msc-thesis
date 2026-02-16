#!/bin/bash

# run with : sbatch src/scorecard.sh 2022-07-01 2022-12-31 stage-c olivia

#SBATCH --job-name=scorecard
#SBATCH --account=nn10090k
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32    # All CPU cores of one Grace-Hopper card
#SBATCH --mem=100G    # Amount of CPU memory
#SBATCH --output=../logs/scorecard_logs/scorecard_%j.out

START=$(printf '%s' "$1")
END=$(printf '%s' "$2")
MODEL=$(printf '%s' "$3")   # models : global, bris, stage-c, stage-d2/3/4
SYSTEM=$(printf '%s' "$4")

CONTAINER_PATH=/cluster/projects/nn10090k/wfrt-anemoi_container/wfrt-anemoi_container.sif
BIND_DIRS=/nird/datapeak/NS10090K,$HOME/anemoi-wfrt,/cluster/projects/nn10090k 
PYSCRIPT="src/scorecard.py --start_date ${START} --end_date ${END} --model ${MODEL} --system ${SYSTEM}"

GPU_LOG_FILE="gpu_utilization_multinode_container.log"
echo "Starting GPU utilization monitoring..."
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 5 > $GPU_LOG_FILE &

echo "Executing apptainer exec --nv --bind $BIND_DIRS $CONTAINER_PATH python $PYSCRIPT"
apptainer exec --nv --bind $BIND_DIRS $CONTAINER_PATH python $PYSCRIPT

echo "Program finished successfully!"
# Stop GPU utilization monitoring
pkill -f "nvidia-smi --query-gpu"

sacct -j $SLURM_JOBID



