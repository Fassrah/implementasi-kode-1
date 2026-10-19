import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. DATA PREPARATION (Mensimulasikan data "PT Maju Bersama")
# Kolom: Jumlah Tagihan, Hari Jatuh Tempo, Metode Pembayaran, Riwayat Telat, Target (1=Late, 0=Not Late)
np.random.seed(42)
data = {
    'InvoiceAmount': np.random.randint(1000, 50000, 500),
    'CreditTerms_Days': np.random.choice([15, 30, 45, 60], 500),
    'PaymentMethod': np.random.choice(['Bank Transfer', 'Credit Card', 'Cash'], 500),
    'PriorLatePayments': np.random.randint(0, 5, 500),
    'Is_Late': np.random.choice([0, 1], 500, p=[0.8, 0.2]) # Data imbalance (banyak yg tidak telat)
}
df = pd.DataFrame(data)

# Mengubah kategori text menjadi angka (One-Hot Encoding)
df_encoded = pd.get_dummies(df, columns=['PaymentMethod'], drop_first=True)

# 2. DEFINISI FITUR (X) DAN TARGET (y)
X = df_encoded.drop('Is_Late', axis=1)
y = df_encoded['Is_Late']

# 3. TRAIN-TEST SPLIT (Membagi data training dan testing 80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. MODELING (Menggunakan Random Forest seperti di demonstrasi KNIME)
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# 5. EVALUASI (Melakukan prediksi & memvalidasi skor)
y_pred = model.predict(X_test)

print("--- HASIL PREDIKSI LATE PAYMENT PT MAJU BERSAMA ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Insight Preskriptif: Mendapatkan fitur yang paling mempengaruhi orang telat bayar
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFitur paling berpengaruh terhadap penunggakan:")
print(feature_importances)