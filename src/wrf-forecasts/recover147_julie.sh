#!/bin/bash
#
# Recover select files from backup tapes
# -----
#
# The preparation to use this script is a manual process, here are the steps to follow.
#
# 1)  Find the tape volume & file no. of the target backup(s) from the backup
#     index database files
#
#     $ for DATE in {21..26}; do grep -h "RDPS/201602${DATE}00" bin/*.csv; done >> data/username-NN.csv
#
# 2)  Remove un-needed text, then sort entries in recover.csv by volume & file no.
#
#     $ sed -i 's/VolumeTag = //' recover.csv
#     $ sed -i 's/File number=//' recover.csv
#     $ sort --field-separator=',' -k 1,1 -k 2n,2 data/username-NN.csv > data/username-NNsorted.csv
#
#     To extract a list of the tapes required:
#
#     $  cat data/username-NNsorted.csv | cut -f1 -d, | uniq > data/username-NNtapes.list
#
# 3)  Load a tape into the workstation drive, and run this script to recover the files
#
#     NOTE: use the test flag (-t) to check your input is correct before runnning the script in anger
#
#     $ bin/recover147.sh -v VOLUME data/username-NNsorted.csv >> log/username.log 2>&1
#
# 2017-01-09 by R. Schigas (rschigas@eos.ubc.ca)
# 2018-07-16 updated by R. Schigas (rschigas@eos.ubc.ca)
#

usage() {
    echo "Restore backed-up files from tape library"
    echo
    echo "NOTE: This script is customized to run from the tape recovery workstation, a23-147"
    echo
    echo "Usage: $0 [-t] [-z] -v VOLUME FILENAME.CSV"
    echo "where: FILENAME.CSV = sorted, comma-delimited CSV file containing the list of"
    echo "                      files to restore (see steps 1-2 of the comments)"
    echo "       -v = only restore files from a single tape, where VOLUME is the volume"
    echo "            tag from the label"
    echo "       -z = (optional) data files are compressed, need to uncompress them"
    echo "       -t = (optional) dry run test mode, commands will be printed instead of"
    echo "            executed"
}
run_cmd() {
    local cmd=${1}
    # Run a command unless in test mode, then print it
    if ${testing}; then
        echo ${cmd}
    else
        eval ${cmd}
    fi
}
log_msg() {
    local msg=${1}
    # Print log message with datetime stamp
    echo "[`/bin/date +"%F %T %Z"`] ${msg}"
}
clean_finish() {
    local msg=${1}
    #run_cmd "/bin/mt -f ${TAPE_DRV} eject"
    rm -f ${PID_FILE}
    log_msg "${msg}"
    echo
}

# Initialize constants
TAPE_DRV="/dev/nst0"
RECOVER_DIR="/"  # using symlinks to /scratch/
PID_FILE="bin/${0}.pid"
OPTIND=1

# Initialize variables
testing=false
one_tape=false
unzip=false
input_file=""
next_tape=""
tape_num=""
is_loaded=false
curr_file=0
next_file=0
increment=0

# Parse optional arguments
while getopts "tv:hz" opt; do
    case "${opt}" in
    v)  one_tape=true
        if [[ -n "${OPTARG}" ]]; then
            curr_tape=${OPTARG}
        else
            echo "You must provide a volume tag; ${0} halting."
            echo
            usage
            exit 1
        fi
        ;;

    z)  unzip=true
        ;;
    t)  testing=true
        ;;
    *)  usage
        exit 0
        ;;
    esac
done
shift $((OPTIND-1))
# Parse positional arguments
if [[ -n "${1}" ]]; then
    input_file=${1}
    if [[ ! -f "${input_file}" ]]; then
        echo "${input_file} does not exist; ${0} halting."
        exit 1
    fi
else
    echo "You must provide a file name; ${0} halting."
    echo
    usage
    exit 1
fi

###  MAIN  ###
set -u
echo
log_msg "Starting ${0} ..."
if pidof -o %PPID -x ${0##*/} > /dev/null; then
    if ! ${testing}; then
        echo "An instance of this script is already running, halting."
        exit -1
    fi
fi
if ${testing}; then
    echo "Running in test mode, commands will be printed instead of executed"
fi

# Get tape drive status
#lib_stat=`/bin/mt -f ${TAPE_LIB} status`
# If tape library is not working ...
#if [[ ${lib_stat} == *"READ ELEMENT STATUS Command Failed"* ]]; then
#    echo "Tape library error, here is the status report:"
#    echo "${lib_stat}"
#    clean_finish "${0} halting."
#    exit 1
#fi

# Rewind tape
run_cmd "/bin/mt -f ${TAPE_DRV} rewind"

# Read each line of the CSV file with the list of files to restore
while IFS=, read -r next_tape next_file backup_file; do
    # If volume tag of file matches volume tag of currently loaded tape ...
    if [[ ${next_tape} == ${curr_tape} ]]; then
        # Check that file has not already been recovered
        if [[ ! -f "${backup_file}.BACKUP.OK" ]]; then
            log_msg "Restoring tape ${next_tape}, file ${next_file} (${backup_file})"
            # Move to correct file number on tape (unless file is at beginning)
            if [[ ${next_file} != 0 ]]; then
                increment=$(( next_file - curr_file ))
            fi
            echo "Queuing tape ..."
            run_cmd "/bin/mt -f ${TAPE_DRV} fsf ${increment}"
            # old (incorect logic?)
            #increment=$(( next_file - curr_file ))
            #if [[ ${curr_file} != 0 ]]; then
            #    increment=$(( increment - 1 ))
            #fi
            #if [[ ${increment} -ne 0 ]]; then
            #   echo "Queuing tape ..."
            #   run_cmd "/bin/mt -f ${TAPE_DRV} fsf ${increment}"
            #fi
            run_cmd "/bin/mt -f ${TAPE_DRV} status | grep number"
            curr_file=${next_file}
            file_id="${backup_file##*/}"
            dir="${backup_file%/*}"
            model_id="${dir##*/}"
            # Extract file from tape
            if ${unzip}; then
                run_cmd "/bin/tar --extract --gunzip --file=${TAPE_DRV} --directory=${RECOVER_DIR} --verbose --wildcards '*_d02_*'"
            else
                run_cmd "/bin/tar --extract --file=${TAPE_DRV} --directory=${RECOVER_DIR} --verbose --wildcards '*_d02_*'"
            fi
            # Set flag if recovery was successful
            if [[ $? -eq 0 ]]; then
                run_cmd "/bin/touch ${backup_file}.BACKUP.OK"
            fi
            run_cmd "/bin/mt -f ${TAPE_DRV} status | grep number"
	    # extract desired vars and copy to nextcloud
	    #run_cmd "touch ${backup_file}/output_${file_id}.nc"
	    #  run_cmd "cdo -L mergetime -apply,-selname,U10,V10,COSALPHA,SINALPHA,T2,RAINNC,RAINC [ ${backup_file}/wrfout_d02_*00 ] ${backup_file}/merged.nc"
	    #run_cmd "cdo -L mergetime $(find ${backup_file} -maxdepth 1 -type f -name 'wrfout_d02_*' | grep -E '_(00|06|12|18):00:00$' | sort) ${backup_file}/output_${file_id}.nc" 
	    #run_cmd "rclone copy ${backup_file}/output_${file_id}.nc wfrt-nextcloud-jcharlet:Documents/WRF-forecasts/${model_id}/ --progress" 
        else
            log_msg "Skipping tape ${next_tape}, file ${next_file} (${backup_file}), already recovered"
        fi
    else
        log_msg "Skipping tape ${next_tape}, file ${next_file} (${backup_file}), not on this volume"
    fi
done < "${input_file}"

if ! ${testing}; then
    /usr/local/bin/slack.sh -T -c "#tape-recovery" -u "tuser@a23-147" -v "${0}" -s "LOW" -t "Tape recovery completed" "Tape volume *${curr_tape}* is done and can be replaced."
fi

# Clean up and finish
for dir in forecasts ibcs results; do
    # Fix permissions to ensure purge script runs correctly
    run_cmd "chmod -R g+w /scratch/${dir} 2>/dev/null"
done
# Rewind tape
run_cmd "/bin/mt -f ${TAPE_DRV} rewind"

clean_finish "${0} is done."
exit 0