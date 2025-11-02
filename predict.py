import os
import json
import joblib
import argparse
import pandas as pd
import numpy as np

# Paths to artifacts
ARTIFACTS_DIR = 'artifacts'
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'best_model.joblib')
SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'scaler.joblib')
ENCODING_MAP_PATH = os.path.join(ARTIFACTS_DIR, 'encoding_map.json')
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, 'features.json')

# Basic checks
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run the training script first.")

# Load artifacts
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(ENCODING_MAP_PATH, 'r', encoding='utf-8') as f:
    encoding_map = json.load(f)
with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
    feature_list = json.load(f)

# Helper to normalize column names
def normalize_cols(df):
    col_map = {}
    for col in df.columns:
        key = col.strip().lower().replace(' ', '').replace('-', '').replace('_', '')
        if key in ('loanstatus',):
            col_map[col] = 'Loan_Status'
        elif key in ('loanid',):
            col_map[col] = 'Loan_ID'
        elif key in ('applicantincome', 'incomeannum'):
            col_map[col] = 'ApplicantIncome'
        elif key in ('coapplicantincome',):
            col_map[col] = 'CoapplicantIncome'
        elif key in ('loanamount',):
            col_map[col] = 'LoanAmount'
        elif key in ('loanamountterm','loanterm'):
            col_map[col] = 'Loan_Amount_Term'
        elif key in ('credithistory','cibilscore'):
            col_map[col] = 'Credit_History'
        elif key in ('propertyarea',):
            col_map[col] = 'Property_Area'
        elif key in ('selfemployed',):
            col_map[col] = 'Self_Employed'
        elif key in ('dependents',):
            col_map[col] = 'Dependents'
        elif key in ('education',):
            col_map[col] = 'Education'
        elif key in ('married',):
            col_map[col] = 'Married'
        elif key in ('gender',):
            col_map[col] = 'Gender'
    if col_map:
        df = df.rename(columns=col_map)
    return df

# Preprocessing for inference
def preprocess(df):
    df = normalize_cols(df)

    # Ensure columns exist
    required_defaults = {
        'Gender': 'Male',
        'Married': 'No',
        'Dependents': '0',
        'Self_Employed': 'No',
        'Property_Area': 'Urban'
    }
    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default

    # Apply encoding map
    for col, mapping in encoding_map.items():
        if col in df.columns:
            # mapping keys are strings
            df[col] = df[col].astype(str).apply(lambda x: mapping.get(x, list(mapping.values())[0]))

    # Dependents
    if 'Dependents' in df.columns:
        df['Dependents'] = df['Dependents'].astype(str).replace('3+', '3')
        df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce').fillna(0)

    # Drop Loan_ID if present
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)

    # Ensure all feature columns exist
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_list].copy()
    # Scale
    X_scaled = scaler.transform(X)
    return df, X_scaled


def main():
    parser = argparse.ArgumentParser(description='Predict loan approval for new applicants (CSV input).')
    parser.add_argument('input_csv', help='Path to input CSV with applicant rows')
    parser.add_argument('--output', '-o', help='Path to save predictions CSV', default=None)
    args = parser.parse_args()

    input_csv = args.input_csv
    out_path = args.output

    df_in = pd.read_csv(input_csv)
    df_processed, X_scaled = preprocess(df_in)

    preds = model.predict(X_scaled)

    # Map back to Y/N if model predicts 1/0
    try:
        # if encoding of Loan_Status was  {'N':0,'Y':1}
        df_processed['Predicted_Loan_Status'] = np.where(preds==1, 'Y', 'N')
    except Exception:
        df_processed['Predicted_Loan_Status'] = preds

    if out_path is None:
        base, ext = os.path.splitext(input_csv)
        out_path = f"{base}_predictions{ext}"

    df_processed.to_csv(out_path, index=False)
    print(f"Predictions saved to: {out_path}")

if __name__ == '__main__':
    main()
