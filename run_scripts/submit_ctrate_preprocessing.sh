#!/bin/bash

# Master script to submit all CT-RATE preprocessing jobs
# Run this as the cxr project account: sudo -u cxr sudosh

echo "Submitting CT-RATE preprocessing jobs..."

# Submit train preprocessing job
echo "Submitting train preprocessing..."
TRAIN_JOB=$(sbatch ctrate_preprocess_train.sh | awk '{print $4}')
echo "Train job ID: $TRAIN_JOB"

# Submit validation preprocessing job
echo "Submitting validation preprocessing..."
VALID_JOB=$(sbatch ctrate_preprocess_valid.sh | awk '{print $4}')
echo "Validation job ID: $VALID_JOB"

# Submit test preprocessing job
echo "Submitting test preprocessing..."
TEST_JOB=$(sbatch ctrate_preprocess_test.sh | awk '{print $4}')
echo "Test job ID: $TEST_JOB"

echo ""
echo "All jobs submitted successfully!"
echo "Monitor progress with: squeue -u \$USER"
echo "Check logs in current directory: ctrate_*_preprocess_*.out"
echo ""
echo "Expected outputs:"
echo "  - /cbica/projects/CXR/data_p/ctrate_train.h5"
echo "  - /cbica/projects/CXR/data_p/ctrate_valid.h5"  
echo "  - /cbica/projects/CXR/data_p/ctrate_test.h5"