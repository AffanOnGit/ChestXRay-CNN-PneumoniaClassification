# Medical Image Classification using Deep Convolutional Neural Networks

This repository contains the source code and LaTeX report for Assignment #1, focusing on classifying chest X-ray images (e.g., Pneumonia vs. Normal) using both a Custom Deep CNN and pre-trained transfer learning architectures (ResNet50, VGG16).

---
## 🛑 User Checkpoint / Next Steps (Resume Here)
As of the last session, the **Custom CNN** has fully finished training and its precision/accuracy metrics have been natively injected into `report/Conference Paper.tex`.

**When you return to this code:**
1. Check your terminal: The `run_experiments.ps1` script should have finished training the **ResNet-50** baseline comparison model.
2. Run the `src/evaluate.py` script on the newly generated ResNet-50 `.keras` file in `saved_models/` to extract its accuracy numbers.
3. Paste those ResNet-50 numbers into Section V of the LaTeX report to complete the quantitative comparison.
4. Open `notebooks/hyperparam_study.ipynb` and press "Run All" to automatically generate the hyperparameter graphs for the report.
---

## Requirements
- **Python 3.10** (Recommended to avoid TensorFlow compatibility issues on Windows)
- `tensorflow>=2.10.0`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `jupyter`

## Setup Instructions

1. **Virtual Environment**:
   It is highly recommended to use a Python 3.10 virtual environment.
   ```bash
   py -3.10 -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Dataset Organization**:
   Download the datasets from Kaggle and merge them into the following structure. Our data loader will automatically use 15% of the `train` folder for validation.
   ```text
   data/
   ├── train/
   │   ├── NORMAL/    (Images from Dataset1/train/NORMAL, Dataset1/val/NORMAL, Dataset2/train/NORMAL)
   │   └── PNEUMONIA/ (Images from Dataset1/train/PNEUMONIA, Dataset1/val/PNEUMONIA, Dataset2/train/PNEUMONIA)
   ├── test/
   │   ├── NORMAL/    (Images from Dataset1/test/NORMAL, Dataset2/test/NORMAL)
   │   └── PNEUMONIA/ (Images from Dataset1/test/PNEUMONIA, Dataset2/test/PNEUMONIA)
   ```

## Execution

### Automated Pipeline
You can run the entire training and default evaluation pipeline via the provided PowerShell script (make sure you are inside your activated virtual environment):
```powershell
.\run_experiments.ps1
```

### Manual Training
```bash
python src/train.py --model custom --batch_size 32 --lr 1e-4 --epochs 50
```
Available models: `custom`, `resnet`, `vgg`.

### Manual Evaluation
```bash
python src/evaluate.py --model_path saved_models/custom_bs32_lr0.0001_do0.3.keras
```

### Hyperparameter Studies
Launch the Jupyter notebook to run the parameterized tests and generate plots:
```bash
jupyter notebook notebooks/hyperparam_study.ipynb
```

## Report Compilation
The IEEE-formatted technical report is located in `report/Conference Paper.tex`. You can compile it using any standard LaTeX distribution (e.g., standard `pdflatex`).
