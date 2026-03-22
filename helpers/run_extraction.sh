#!/bin/bash

# --- Configuration ---
TOTAL_PARTS=1000
PYTHON_SCRIPT="create_slice_dataset_oo.py"

echo "Starting sequential extraction for $TOTAL_PARTS parts..."

# Loop from 0 to TOTAL_PARTS - 1
for (( i=0; i<$TOTAL_PARTS; i++ ))
do
    echo "========================================"
    echo "Processing Part $i of $((TOTAL_PARTS-1))..."
    echo "========================================"
    
    # Execute the python script with the current index
    python $PYTHON_SCRIPT --total_parts $TOTAL_PARTS --part_index $i
    
    # Catch errors: if the Python script fails, stop the bash loop
    if [ $? -ne 0 ]; then
        echo "Error: Extraction failed at part $i. Aborting the sequence."
        exit 1
    fi
    
    echo "Part $i finished successfully."
done

echo "========================================"
echo "All $TOTAL_PARTS parts have been extracted and saved to disk!"