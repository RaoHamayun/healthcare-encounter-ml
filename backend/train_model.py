import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/processed/features.csv")

features = [
    'age',
    'is_inpatient',
    'is_ambulatory',
    'is_wellness',
    'clinical_burden'
]

X = df[features]
y = df['has_condition']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

joblib.dump(model, "ml_models/random_forest.pkl")
