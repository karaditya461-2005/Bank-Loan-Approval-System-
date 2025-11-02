Loan Approval Prediction
What this ML model does
This project implements a supervised machine learning pipeline to predict whether a bank should approve or reject a loan application. The script performs exploratory data analysis (EDA), data preprocessing (imputation and encoding), model training (multiple classifiers), and evaluation. The trained models are evaluated using accuracy, precision, recall and F1-score. The project selects the best-performing model (by F1-score) for final analysis.

The code included with this project trains and compares the following models:

Logistic Regression
Decision Tree Classifier
Random Forest Classifier (selected as best model in typical runs)
K-Nearest Neighbors
In typical runs provided by the script, the Random Forest model often achieves the best balance of precision and recall for this loan approval dataset.

How the model is applied to the loan approval problem
Problem statement:

Given applicant information (demographic, income, credit history and loan details), predict a binary outcome: Approved (Y) or Rejected (N).
Approach summary:

Data ingestion: the script reads a CSV (expected path: data/train.csv) with one row per loan application and columns such as applicant income, loan amount, credit score/history, education, marital status, property area, etc.

Exploratory data analysis (EDA): the script prints basic dataset summaries and creates visualizations (saved as PNG files) to show distributions and relationships (e.g., Education vs Loan Status, Income distribution, Marital status, Property area, Loan amount, Credit history distribution).

Preprocessing:

Missing values: numerical columns (e.g., LoanAmount) are filled with median values; Loan_Amount_Term and Credit_History use mode imputation; categorical columns use mode.
Encoding: categorical features (Gender, Married, Education, Self_Employed, Property_Area) are label-encoded, and the target (Loan_Status) is converted to binary Y/N.
Scaling: features are standardized with StandardScaler before using models that require scaling (Logistic Regression, KNN).
Modeling & evaluation:

The dataset is split into training and test sets (default 80/20, stratified by target).
Models are trained and predictions are made on the test set.
Performance metrics reported: Accuracy, Precision, Recall, F1-Score. The script uses F1-score as the primary selection metric because it balances precision and recall and is robust to moderate class imbalance.
The script also outputs a classification report, confusion matrix, and (for tree-based models) feature importance.
Output: visualizations and evaluation charts are saved to the project root, e.g.:

eda_visualizations.png
additional_visualizations.png
model_comparison.png
confusion_matrix.png
feature_importance.png (if applicable)
Files and structure
loan_prediction.py — main script (EDA, preprocessing, modeling, evaluation). The script has a synthetic-data fallback so it can run even when an expected CSV isn't present.
data/train.csv — expected dataset location. If the file is missing, the script will generate a small synthetic dataset and save it here for demo runs.
Output PNGs — visualizations and evaluation charts (listed above).
How to run
From the project root (Windows PowerShell), run:

python .\loan_prediction.py
This will:

read data/train.csv if present (the script adapts to some common column name variants),
otherwise generate a small synthetic dataset at data/train.csv and continue,
run EDA, preprocessing, model training, and evaluation,
save chart PNG files in the project folder.
Dependencies
Install the required libraries (recommended to use a virtual environment):

pip install pandas numpy matplotlib seaborn scikit-learn
Notes & next steps
If you have a specific dataset you want used instead of the synthetic fallback, place it at data/train.csv. The script normalizes a number of common header variants, but for best results use column names similar to: Loan_ID, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Loan_Status.

For production usage consider:

Saving the trained best model to disk (e.g., using joblib or pickle).
Implementing proper cross-validation and hyperparameter tuning (GridSearchCV or randomized search).
Adding unit tests for preprocessing and model-prediction code paths.
Adding an inference wrapper (Flask/FastAPI) to serve predictions.
If you want, I can (a) add a requirements.txt, (b) save the trained Random Forest as model.joblib, or (c) add a minimal inference script. Tell me which you prefer.