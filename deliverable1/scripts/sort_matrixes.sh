#!/bin/bash

# Path to your sort_mtx executable
SORT_BIN=./GPU_Deliverables/deliverable1/scripts/sort

# Folder containing .mtx files
DATA_DIR=./data


# Check if executable exists
if [ ! -x "$SORT_BIN" ]; then
    gcc -O3 -march=native -o ./GPU_Deliverables/deliverable1/scripts/sort ./GPU_Deliverables/deliverable1/scripts/sort.c
fi

# Process each .mtx file
for file in "$DATA_DIR"/*.mtx; do
    echo "Sorting $file..."
    "$SORT_BIN" "$file"
    # Remove original file
    rm "$file"
done