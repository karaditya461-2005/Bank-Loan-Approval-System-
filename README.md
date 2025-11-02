<<<<<<< HEAD
# Loan Approval Prediction System 🏦

This project implements a machine learning system to predict bank loan approvals based on applicant information such as income, credit history, and other factors.

## Project Structure 📂

```
loan_approval_project/
│
├── data/
│   └── train.csv                  # Training dataset
│
├── outputs/                       # Generated visualizations
│   ├── eda_visualizations.png    
│   ├── additional_visualizations.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── artifacts/                     # Saved model & preprocessing objects
│   ├── best_model.joblib         # Trained Random Forest
│   ├── scaler.joblib             # StandardScaler
│   ├── encoding_map.json         # Category encodings
│   └── features.json             # Feature column list
│
├── loan_prediction.py            # Main training script
├── predict.py                    # Inference script
├── requirements.txt              # Dependencies list
└── README.md                     # This file
```

## Quick Start 🚀

1. Install dependencies:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Train model (creates visualizations & saves model):
```powershell
python loan_prediction.py
```

3. Make predictions on new data:
```powershell
python predict.py path/to/new_applications.csv
```

## Model Details 🤖

The script trains and compares multiple classifiers:
- Logistic Regression
- Decision Tree
- Random Forest (typically best performer)
- K-Nearest Neighbors

Features used:
- Credit History (strongest predictor)
- Loan Amount & Term
- Income (Applicant & Co-applicant)
- Education
- Employment Type
- Assets (Residential, Commercial, etc.)
- Other Demographics

Performance metrics (typical):
- Accuracy: ~97%
- Precision: ~98%
- Recall: ~98%
- F1-Score: ~98%

## Outputs 📊

The training script (`loan_prediction.py`) generates several visualizations in the `outputs/` directory:

- `eda_visualizations.png`: Key feature distributions and relationships
- `additional_visualizations.png`: Demographics vs. loan approval charts
- `model_comparison.png`: Performance comparison across models
- `confusion_matrix.png`: Detailed view of predictions vs. actuals
- `feature_importance.png`: Feature importance ranking (for tree models)

## Making Predictions 🎯

The `predict.py` script loads the trained model and makes predictions on new applicant data:

1. Prepare a CSV with applicant information
2. Run: `python predict.py your_file.csv`
3. Get predictions in `your_file_predictions.csv`

Notes:
- Column names are flexible (e.g., "income_annum" or "ApplicantIncome")
- Missing values are handled with reasonable defaults
- See example CSVs in `data/` for reference format

## Dependencies 📦

Main requirements (see `requirements.txt` for versions):
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

## Future Improvements 🔄

Potential enhancements:
- [ ] Add cross-validation
- [ ] Implement hyperparameter tuning
- [ ] Create web API for predictions
- [ ] Add more error handling in predict.py
- [ ] Expand testing coverage
- [ ] Add model monitoring
=======
# Bank-Loan-Approval-System-
A Machine Learning model that predicts bank loan approval using applicant details
>>>>>>> 5fb9749948d1a7d813ce3fc1a7cca53421bef45f
