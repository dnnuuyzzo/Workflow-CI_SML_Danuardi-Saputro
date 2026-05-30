import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

def load_data(file_path):
    print(f"Loading data dari {file_path}")
    return pd.read_csv(file_path)

def main():
    # Mengaktifkan MLflow autologging
    mlflow.autolog()
    
    # Menyiapkan path data
    # Dataset preprocessed_churn.csv berada di folder yang sama (MLProject)
    data_path = "preprocessed_churn.csv"
    
    # Memuat data
    try:
        df = load_data(data_path)
    except FileNotFoundError:
        print(f"File tidak ditemukan di {data_path}. Pastikan path dataset benar.")
        return
    
    # Memisahkan fitur dan target
    # 'Churn' adalah target (telah di-encode menjadi 0 dan 1 pada tahap eksperimen)
    if 'Churn' not in df.columns:
        print("Kolom 'Churn' tidak ditemukan di dataset.")
        return
        
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Splitting data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Jumlah data latih: {X_train.shape[0]}")
    print(f"Jumlah data uji: {X_test.shape[0]}")
    
    # Mengatur nama eksperimen di MLflow (Opsional tapi direkomendasikan agar rapi)
    mlflow.set_experiment("Telco_Customer_Churn_Basic_Model")
    
    # Memulai MLflow run
    with mlflow.start_run(run_name="RandomForest_Basic"):
        print("Memulai pelatihan model RandomForestClassifier...")
        
        # Inisiasi model dasar tanpa hyperparameter tuning
        model = RandomForestClassifier(random_state=42)
        
        # Pelatihan model (disinilah autolog akan merekam model, parameter, dan metrik pelatihan)
        model.fit(X_train, y_train)
        
        # Evaluasi tambahan pada data uji (opsional, karena autolog mungkin sudah mencatatnya, 
        # tapi eksplisit lebih baik untuk report CLI)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print("\n=== HASIL EVALUASI PADA DATA UJI ===")
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nModel telah selesai dilatih dan dicatat oleh MLflow autolog.")
        print("Jalankan 'mlflow ui' di terminal untuk melihat artefak dan metrik di browser.")

if __name__ == "__main__":
    main()
