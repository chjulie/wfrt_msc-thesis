#!/bin/sh -l
### Example usage: sbatch src/scorecard_fir.sh 2023-01-01 2023-12-31 nwp_reg fir

#SBATCH --account=rrg-rstull
#SBATCH --job-name=scorecard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 # Allows us to set optimal "threads" value in our configuration.json file to have multi-threaded data processing
#SBATCH --mem-per-cpu=64G ##total requested cpu memory => must be large enough for anemoi-datasets create!! 64G = 10 days, 
#SBATCH --time=06:00:00
#SBATCH --output=./logs/scorecard_logs/scorecard_%j.out

source .venv/bin/activate
# prepare log file
START=$(printf '%s' "$1")
END=$(printf '%s' "$2")
MODEL=$(printf '%s' "$3")   # models : global, bris, stage-c, stage-d2/3/4
SYSTEM=$(printf '%s' "$4")

uv run src/scorecard.py --start_date "${START}" --end_date "${END}" --model "${MODEL}" --system "${SYSTEM}"
echo " ✅  Script finished."