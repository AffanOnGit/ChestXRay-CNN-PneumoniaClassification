#!/bin/bash

echo "=========================================================="
echo " Starting Chest X-Ray CNN Assignment Experiments Pipeline "
echo "=========================================================="

echo "1. Installing dependencies"
pip install -r requirements.txt

# Create necessary directories if they don't exist
mkdir -p data/train data/val data/test
echo "NOTE: Please ensure you manually download the Kaggle Chest X-Ray datasets and place them into the data/ directory before running the experiments."

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "2. Training Custom CNN Baseline"
python src/train.py --model custom --batch_size 32 --lr 1e-4 --epochs 50

echo "3. Training Pre-trained ResNet50 for Comparison"
python src/train.py --model resnet --batch_size 32 --lr 1e-4 --epochs 50

echo "4. Running Evaluation on Custom CNN"
# Assuming standard naming convention from the saving step
# Change to exact filename dynamically later or pass via arg
LATEST_CUSTOM=$(ls -t saved_models/custom_*.keras | head -n 1)
if [[ -f "$LATEST_CUSTOM" ]]; then
    python src/evaluate.py --model_path "$LATEST_CUSTOM"
fi

echo "All complete! Please check the 'plots' and 'logs' directories for visualization output."
