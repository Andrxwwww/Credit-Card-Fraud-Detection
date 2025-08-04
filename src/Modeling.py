import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
from xgboost import XGBClassifier
import joblib

def train_and_save_models(featured_csv_path, models_dir='models'):
    if not os.path.exists(featured_csv_path):
        raise FileNotFoundError(f"Featured data file not found at {featured_csv_path}")

    os.makedirs(models_dir, exist_ok=True)
    print(f"Loading featured data from {featured_csv_path}...")
    df = pd.read_csv(featured_csv_path)

    if 'is_fraud' not in df.columns:
        raise ValueError("Label column 'is_fraud' not found in data.")

    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=100, n_jobs=-1, class_weight='balanced', random_state=42),
        'lightgbm': LGBMClassifier(
            n_estimators=100, is_unbalance=True, n_jobs=-1, random_state=42),
        'xgboost': XGBClassifier(
            n_estimators=100, n_jobs=-1, use_label_encoder=False,
            eval_metric='logloss', scale_pos_weight=(len(y_train) / y_train.sum()), random_state=42),
    }

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        print(f"Evaluation report for {name}:")
        print(classification_report(y_test, y_pred, digits=4))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
        if y_proba is not None:
            print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

        model_path = os.path.join(models_dir, f"fraud_{name}.joblib")
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")
