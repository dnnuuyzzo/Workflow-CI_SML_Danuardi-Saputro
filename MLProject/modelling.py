import os
import mlflow
import pandas as pd
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data(file_path):
    print(f"Loading data dari {file_path}")
    return pd.read_csv(file_path)

def main():
    mlflow.autolog()
    data_path = "Telco-Customer-Churn_Dataset_preprocessing.csv"

    try:
        df = load_data(data_path)
    except FileNotFoundError:
        print(f"File tidak ditemukan di {data_path}. Pastikan path dataset benar.")
        return
    
    if 'Churn' not in df.columns:
        print("Kolom 'Churn' tidak ditemukan di dataset.")
        return
        
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Jumlah data latih: {X_train.shape[0]}")
    print(f"Jumlah data uji: {X_test.shape[0]}")
    
    with mlflow.start_run():
        print("Memulai pelatihan model...")

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print("\nHASIL EVALUASI PADA DATA UJI".center(25))
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nModel telah selesai dilatih dan dicatat oleh MLflow autolog.")

if __name__ == "__main__":
    main()
