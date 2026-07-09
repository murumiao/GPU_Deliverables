#!/bin/bash

DATA_DIR="./GPU_Deliverables/deliverable1/data"

for FILE in "$DATA_DIR"/*; do
    # Skip if not a regular file
    [ -f "$FILE" ] || continue

    MODES_CSR=(0 1 2 3)
    for MODE in "${MODES_CSR[@]}"; do
        sbatch ./GPU_Deliverables/deliverable1/scripts/csr_run_on_cluster.sh "$FILE" "$MODE"
    done


    MODES_COO=(0 1)

    for MODE in "${MODES_COO[@]}"; do
        sbatch ./GPU_Deliverables/deliverable1/scripts/coo_run_on_cluster.sh "$FILE" "$MODE"
    done
    sbatch ./GPU_Deliverables/deliverable1/scripts/cpu_run_on_cluster.sh "$FILE"
done
