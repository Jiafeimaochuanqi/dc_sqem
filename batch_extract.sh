#!/bin/bash

if [[ "$CONDA_DEFAULT_ENV" != "dc_sqem" ]]; then
    # Initialize conda (only when inactive).
    source "$CONDA_PREFIX/etc/profile.d/conda.sh"
    if ! conda activate dc_sqem; then
        echo "Failed to activate Conda environment dc_sqem. Exiting..."
        exit 1
    fi
else
    echo "Already in conda environment: $CONDA_DEFAULT_ENV"
fi

# Check whether the running script exists.
if [ ! -f "run_mat_pcl_sqef.py" ]; then
    echo "Training script run_mat_pcl_sqef.py not found. Exiting..."
    exit 1
fi

# ============== params ==============
DATASETS_DIR="/mnt/WindowsI/datasets/scaled/ABC"
# DATASETS_DIR="/mnt/WindowsI/datasets/scaled/noncad"
#EXPERIMENTS_DIR="/mnt/WindowsI/datasets/network/ABC"
INPUT_DIR="/mnt/WindowsI/cloud_patch/ABC"
# INPUT_DIR="/mnt/WindowsI/cloud_patch/noncad"
OUTPUT_DIR="/mnt/WindowsI/octree_patch_cpp/ABC"
# OUTPUT_DIR="/mnt/WindowsI/octree_group/noncad"
# process config
MAX_CONCURRENT=4           # Maximum number of concurrent processes
CURRENT_JOBS=0             # Current running process count
PID_FILE="./logs/run_mat_pcl_sqef_pids.log"  # PID file

# ============== functions ==============
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
                echo "process $train_pid (task: $name) had finished!"
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
mkdir -p "./logs"
> "$PID_FILE"

#Count all .obj files in the directory.
total_files=$(ls -1 "$DATASETS_DIR"/*.obj 2>/dev/null | wc -l)
invalid_count=0
skip_count=0
processed_count=0

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

    INPUT_PATH="$INPUT_DIR/$NAME"
    INPUT_NPZ_FILE="$INPUT_PATH/$NAME.sampling.npz"
    if [ ! -f "$INPUT_NPZ_FILE" ]; then
        #Not completed, increment count by 1.
        echo "$INPUT_NPZ_FILE Input file does not exist, skipping calculation."
        invalid_count=$((invalid_count + 1))
        continue
    fi

    #Create the output directory.
    OUTPUT_PATH="$OUTPUT_DIR/$NAME"
    mkdir -p "$OUTPUT_PATH"
    # TARGET_PKL_FILE="$OUTPUT_PATH/$NAME.depth7.sqef.pkl"
    TARGET_OBJ_FILE="$OUTPUT_PATH/$NAME.depth7.tri.sqef.obj"
    TARGET_NPZ_FILE="$OUTPUT_PATH/$NAME.depth7.sqef.npz"
    
    #Check whether both files exist at the same time
    # if [ -f "$TARGET_PKL_FILE" ] && [ -f "$TARGET_NPZ_FILE" ]; then
    if [ -f "$TARGET_OBJ_FILE" ] && [ -f "$TARGET_NPZ_FILE" ]; then
        echo "$NAME output file $TARGET_OBJ_FILE and $TARGET_NPZ_FILE is existed,skip it."
        skip_count=$((skip_count + 1))
        continue
    fi
    
    #Wait for an available process slot.
    wait_for_free_slot
    
    #start computation
    nohup python run_mat_pcl_sqef.py \
        --dataset "$NAME" \
        --input_dir "$INPUT_PATH" \
        --max_depth 7 \
        --debug 1 \
        --output_dir "$OUTPUT_PATH" \
        > "$OUTPUT_PATH/extract_$NAME.log" 2>&1 &
    
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

echo "All reconstruction calculation tasks have been completed."
echo "There are a total of ${total_files} .obj files."
echo "There are still ${invalid_count} training or projection tasks not completed."
echo "Skip ${skip_count} reconstruction tasks."
echo "Execute $processed_count reconstruction tasks."

#Update the PID file and the current job count.
> "$PID_FILE"

# nohup ./batch_extract.sh > logs/script_ABC_extract.log 2>&1 &
#nohup ./batch_extract.sh > logs/script_noncad_extract.log 2>&1 &
#pkill -f "batch_extract.sh"
#find . -name "*.log" -exec rm {} \;
#du -sh .