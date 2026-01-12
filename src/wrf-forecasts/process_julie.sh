#!/bin/bash
FILE_ID=$(printf '%s' "$1")
MODEL_ID=WAC00WG-01
BACKUP_FILE=/archives/forecasts/${MODEL_ID}/${FILE_ID}

eval "touch ${BACKUP_FILE}/merged.nc"
#eval "cdo -L mergetime $(find ${BACKUP_FILE} -maxdepth 1 -type f -name 'wrfout_d02_*' | grep -E '_(00|06|12|18):00:00$' | sort |sed 's/^/"/;s/$/"/') ${BACKUP_FILE}/merged.nc" 

# Collect all wrfout files with the desired hours into an array
files=()
while IFS= read -r file; do
    files+=("$file")
done < <(find "$BACKUP_FILE" -maxdepth 1 -type f -name 'wrfout_d02_*' \
        | grep -E '_(00|06|12|18):00:00$' \
        | sort)

# Check if we found any files
if [ ${#files[@]} -eq 0 ]; then
    echo "No files found to merge in $BACKUP_FILE"
    exit 1
fi

# Run cdo mergetime on all files
cdo -L mergetime "${files[@]}" "$BACKUP_FILE/merged.nc"

eval "python bin/process_julie.py --model ${MODEL_ID} --id ${FILE_ID}"
eval "rclone copy ${BACKUP_FILE}/wrfout_d02_processed_${FILE_ID}.nc wfrt-nextcloud-jcharlet:Documents/WRF-forecasts/${MODEL_ID}/ --progress" 
echo "Script finished successfully !"