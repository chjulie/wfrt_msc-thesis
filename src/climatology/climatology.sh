#!/bin/sh -l

#SBATCH --account=rrg-rstull
#SBATCH --job-name=dataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=64G
#SBATCH --time=00:10:00
#SBATCH --output=../../logs/climatology.out
#SBATCH --error=../../logs/climatology.err

source .venv/bin/activate
uv run compute_climatology.py