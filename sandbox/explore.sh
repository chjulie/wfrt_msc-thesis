#!/bin/sh -l
### Example usage: sbatch src/evaluate.sh 2023-01-01 2023-01-05 nwp_reg

#SBATCH --account=rrg-rstull
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1 # Allows us to set optimal "threads" value in our configuration.json file to have multi-threaded data processing
#SBATCH --mem-per-cpu=64G ##total requested cpu memory => must be large enough for anemoi-datasets create!! 64G = 10 days, 
#SBATCH --time=00:30:00
#SBATCH --output=../logs/explore_inference_%j.out

source ../.venv/bin/activate
# prepare log file
FIELD=$(printf '%s' "$1")

uv run 06_explore_inference_file.py --field "${FIELD}" 
echo " ✅  Script finished."
