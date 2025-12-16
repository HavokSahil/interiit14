#!/bin/bash
# Script to generate training plots

echo "============================================================"
echo "GENERATING TRAINING PLOTS"
echo "============================================================"
echo ""
echo "This will:"
echo "  1. Train CQL, IQL, BCQ models (30 epochs)"
echo "  2. Collect training data every 3 epochs"
echo "  3. Generate 4 plots from real data"
echo ""
echo "Estimated time: 15-30 minutes"
echo ""

cd /home/rishu/Arista/production_package

# Check if dataset exists
if [ ! -f "data/rrm_dataset_expanded.h5" ]; then
    echo "Generating dataset first..."
    python generate_dataset.py --samples 10000 --output data/rrm_dataset_expanded.h5
fi

# Run training and plot generation
python generate_training_plots.py \
    --dataset data/rrm_dataset_expanded.h5 \
    --epochs 30 \
    --eval-freq 3 \
    --output-dir plots

echo ""
echo "============================================================"
echo "✓ DONE! Check plots/ directory for results"
echo "============================================================"
