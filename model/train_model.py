import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dữ liệu
df = pd.read_csv("data/gia_nha_hcm_clean.csv")

# Tách feature và target
X = df[["dien_tich", "gia_m2", "so_phong_ngu", "so_phong_tam"]]
y = df["tong_gia"]

# Tách train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# Lưu mô hình
joblib.dump(model, "model/linear_model.pkl")
print("✅ Mô hình đã được huấn luyện và lưu thành công.")
