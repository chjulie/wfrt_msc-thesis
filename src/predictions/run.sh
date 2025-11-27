#!/bin/sh -l

#SBATCH --nodes=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=4G        # total requested cpu memory 
#SBATCH --time=1:00:00
#SBATCH --output=./output_%j.out

source ../../.venv/bin/activate
uv run inference_aifs_single-v1.py --date 2025-11-26T06:00:00