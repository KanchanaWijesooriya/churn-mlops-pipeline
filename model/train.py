from google.cloud import bigquery
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from google.cloud import storage

PROJECT_ID = "churn-mlops-pipeline"
DATASET = "churn_data"
TABLE = "cleaned_telco_churn"
BUCKET_NAME = "churn-mlops-pipeline-data-ckw"

def load_data():
    client = bigquery.Client(project=PROJECT_ID)
    query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
    """
    df = client.query(query).to_dataframe()
    return df

def preprocess(df):
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    X = df_encoded.drop(columns=["Churn"])
    y = df_encoded["Churn"].astype(int)

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }
    return metrics

def save_model(model, feature_columns):
    local_path = "model/artifacts/model.joblib"
    joblib.dump({"model": model, "feature_columns": feature_columns}, local_path)
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob("models/model.joblib")
    blob.upload_from_filename(local_path)
    print(f"Model uploaded to gs://{BUCKET_NAME}/models/model.joblib")

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    model = train_model(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    print("Metrics:", metrics)
    save_model(model, feature_columns=X_train.columns.tolist())