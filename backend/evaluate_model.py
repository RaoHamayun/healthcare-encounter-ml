import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("data/processed/features.csv")
model = joblib.load("ml_models/random_forest.pkl")

features = [
    'age',
    'is_inpatient',
    'is_ambulatory',
    'is_wellness',
    'clinical_burden'
]

X = df[features]
y = df['has_condition']

y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))
print("ROC AUC:", roc_auc_score(y, y_prob))
