Write-Host "=========================================================="
Write-Host " Starting Chest X-Ray CNN Assignment Experiments Pipeline "
Write-Host "=========================================================="

Write-Host "1. Installing dependencies"
pip install -r requirements.txt

# Create necessary directories if they don't exist
New-Item -ItemType Directory -Force -Path "data\train"
New-Item -ItemType Directory -Force -Path "data\val"
New-Item -ItemType Directory -Force -Path "data\test"
Write-Host "NOTE: Please ensure you manually download the Kaggle Chest X-Ray datasets and place them into the data\ directory before running the experiments."

$env:PYTHONPATH = "$PWD"

Write-Host "2. Training Custom CNN Baseline"
python src\train.py --model custom --batch_size 32 --lr 1e-4 --epochs 50

Write-Host "3. Training Pre-trained ResNet50 for Comparison"
python src\train.py --model resnet --batch_size 32 --lr 1e-4 --epochs 50

Write-Host "4. Running Evaluation on Custom CNN"
$LatestCustom = Get-ChildItem -Path "saved_models\custom_*.keras" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($LatestCustom) {
    python src\evaluate.py --model_path $LatestCustom.FullName
}

Write-Host "All complete! Please check the 'plots' and 'logs' directories for visualization output."
