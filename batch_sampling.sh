#!/bin/bash

if [[ "$CONDA_DEFAULT_ENV" != "dc_sqem" ]]; then
    ## Initialize conda (only when inactive).
    source "$CONDA_PREFIX/etc/profile.d/conda.sh"
    if ! conda activate dc_sqem; then
        echo "Failed to activate Conda environment dc_sqem. Exiting..."
        exit 1
    fi
else
    echo "Already in conda environment: $CONDA_DEFAULT_ENV"
fi

#Check whether the running script exists.
if [ ! -f "run_mesh_sampling.py" ]; then
    echo "Training script run_mesh_sampling.py not found. Exiting..."
    exit 1
fi




# ============== params ==============
DATASETS_DIR="/mnt/WindowsH/datasets/scaled/ABC"
#DATASETS_DIR="/mnt/WindowsH/datasets/scaled/noncad"
OUTPUT_DIR="/mnt/WindowsI/cloud_patch/ABC"
#OUTPUT_DIR="/mnt/WindowsI/cloud_patch/noncad"
# process config
MAX_CONCURRENT=4           # Maximum number of concurrent processes
CURRENT_JOBS=0             # Current running process count
PID_FILE="./logs/running_sampling_pids.log"  # PID file

# ============== function ==============
wait_for_free_slot() {
    while [ $CURRENT_JOBS -ge $MAX_CONCURRENT ]; do
        #Check and clean up completed processes.
        update_job_status
        
        if [ $CURRENT_JOBS -ge $MAX_CONCURRENT ]; then
            echo "Maximum number of processes reached ($MAX_CONCURRENT), waiting ..."
            sleep 10
        fi
    done
}

update_job_status() {
    if [ -f "$PID_FILE" ]; then
        local running_jobs=0
        local temp_file=$(mktemp)
        
        while read -r line; do
            read -r name train_pid <<< "$line"
            if kill -0 "$train_pid" 2>/dev/null; then
                echo "$name $train_pid" >> "$temp_file"
                running_jobs=$((running_jobs + 1))
            else
                echo "process $train_pid (task: $name) has finished!"
            fi
        done < "$PID_FILE"
        
        #Update PID file and current job count.
        if [ -f "$temp_file" ]; then
            mv "$temp_file" "$PID_FILE"
        else
            > "$PID_FILE"
        fi
        
        CURRENT_JOBS=$running_jobs
    else
        CURRENT_JOBS=0
    fi
}

# ============== main process ==============
#Clean up old records.
mkdir -p ./logs
> "$PID_FILE"

#Count all .obj files in the directory.
total_files=$(ls -1 "$DATASETS_DIR"/*.obj 2>/dev/null | wc -l)
skip_count=0
processed_count=0
skip_feature=0
#Traverse the dataset.
for file in "$DATASETS_DIR"/*.obj; do
    #Check if the file actually exists.
    if [[ ! -f "$file" ]]; then
        echo "No .obj files found in $DATASETS_DIR"
        break
    fi
    
    #Parse the filename.
    FILENAME=${file##*/}
    NAME=${FILENAME%.obj}

    #Create the output directory.
    OUT_PATH="$OUTPUT_DIR/$NAME"
    mkdir -p "$OUT_PATH"
    TARGET_NPZ_FILE="$OUT_PATH/$NAME.sampling.npz"
    
    #Check whether file exist
    if [ -f "$TARGET_NPZ_FILE" ]; then
        echo "$NAME output file $TARGET_NPZ_FILE is existed,skip it."
        skip_count=$((skip_count + 1))
        continue
    fi
    
    #Wait for an available process slot.
    wait_for_free_slot
    
    #start computation
    nohup python run_mesh_sampling.py \
        --dataset "$NAME" \
        --datasets_dir "$DATASETS_DIR" \
        --output_dir "$OUT_PATH" \
        --skip_feature "$skip_feature" \
        > "$OUT_PATH/sampling_$NAME.log" 2>&1 &
    
    TRAIN_PID=$!
    CURRENT_JOBS=$((CURRENT_JOBS + 1))
    
    echo "Start computation: $NAME (PID=$TRAIN_PID) [Current number of processes: $CURRENT_JOBS/$MAX_CONCURRENT]"
    echo "$NAME $TRAIN_PID" >> "$PID_FILE"
    processed_count=$((processed_count + 1))
    
    #A short delay to avoid resource contention.
    sleep 2
done

#Wait for all remaining tasks to complete.
echo "All tasks have been started, waiting for completion..."
while [ $CURRENT_JOBS -gt 0 ]; do
    update_job_status
    if [ $CURRENT_JOBS -gt 0 ]; then
        echo "wait $CURRENT_JOBS processes to finish..."
        sleep 10
    fi
done

echo "All sampling tasks have been completed."
echo "There are a total of ${total_files} .obj files."
echo "Skip ${skip_count} reconstruction tasks."
echo "Execute $processed_count sampling tasks."

#Update the PID file and the current job count.
> "$PID_FILE"

#mkdir -p logs &&nohup ./batch_sampling.sh > logs/script_abc_sampling.log 2>&1 &
#pkill -f "batch_sampling.sh"
#find . -name "*.log" -exec rm {} \;
#du -sh .