import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Đọc dữ liệu đã clean và encode
df = pd.read_csv("data/gia_nha_hcm_clean.csv")
print("🧾 Tên các cột trong dữ liệu:")
print(df.columns)

# Biến đầu vào: tất cả trừ cột mục tiêu
features = [col for col in df.columns if col != "tong_gia"]
X = df[features]
y = df["tong_gia"]

# Huấn luyện model
model = LinearRegression()
model.fit(X, y)

# Lưu model
with open("model/linear_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Lưu danh sách cột features
with open("model/feature_columns.pkl", "wb") as f:
    pickle.dump(features, f)

print("✅ Đã huấn luyện xong và lưu model.")
