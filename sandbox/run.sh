#!/bin/sh -l
### Example usage sbatch src/data_processing/dataset.sh 2020-06-01 2020-06-03 training

#SBATCH --account=rrg-rstull
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8 # Allows us to set optimal "threads" value in our configuration.json file to have multi-threaded data processing
#SBATCH --mem-per-cpu=64G ##total requested cpu memory => must be large enough for anemoi-datasets create!! 64G = 10 days, 
#SBATCH --time=00:10:00
#SBATCH --output=/home/juchar/wfrt_msc-thesis/logs/run_%j.out

source .venv/bin/activate
uv run sandbox/05-grid_interpolation.py