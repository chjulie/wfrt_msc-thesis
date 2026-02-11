#!/bin/sh -l
### Example usage: sbatch src/scorecard.sh 2023-01-01 2023-02-01 nwp_reg

#SBATCH --account=rrg-rstull
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1 # Allows us to set optimal "threads" value in our configuration.json file to have multi-threaded data processing
#SBATCH --mem-per-cpu=64G ##total requested cpu memory => must be large enough for anemoi-datasets create!! 64G = 10 days, 
#SBATCH --time=03:00:00
#SBATCH --output=./logs/scorecard_%j.out

source .venv/bin/activate
# prepare log file
START_DATE=$(printf '%s' "$1")
END_DATE=$(printf '%s' "$2")
MODEL=$(printf '%s' "$3")

uv run src/scorecard.py --start_date "${START_DATE}" --end_date "${END_DATE}" --model "${MODEL}" 
echo " ✅  Script finished."
