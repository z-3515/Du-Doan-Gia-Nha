import streamlit as st
import pickle
import numpy as np

# Load model
with open("model/linear_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature columns
with open("model/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# 🔍 Trích danh sách vị trí từ feature_columns
vi_tri_columns = [col for col in feature_columns if col.startswith("vi_tri_")]
vi_tri_options = [col.replace("vi_tri_", "") for col in vi_tri_columns]

st.title("🔮 Dự đoán giá nhà tại TP.HCM")

st.markdown("Nhập thông tin để dự đoán tổng giá trị căn nhà (đơn vị: **triệu đồng**)")

# 👉 Nhập thông tin
dien_tich = st.number_input("Diện tích (m2)", min_value=10.0, max_value=1000.0, step=1.0)
so_phong_ngu = st.number_input("Số phòng ngủ", min_value=0, max_value=10, step=1)
so_phong_tam = st.number_input("Số phòng tắm", min_value=0, max_value=10, step=1)
vi_tri = st.selectbox("Vị trí", vi_tri_options)

# 👉 Dự đoán khi ấn nút
if st.button("📈 Dự đoán giá"):
    # Tạo input vector với thứ tự như feature_columns
    input_data = []

    for col in feature_columns:
        if col == "dien_tich":
            input_data.append(dien_tich)
        elif col == "so_phong_ngu":
            input_data.append(so_phong_ngu)
        elif col == "so_phong_tam":
            input_data.append(so_phong_tam)
        elif col.startswith("vi_tri_"):
            input_data.append(1 if col == f"vi_tri_{vi_tri}" else 0)
        else:
            input_data.append(0)  # Phòng khi dư cột

    # Chuyển thành numpy array
    input_array = np.array(input_data).reshape(1, -1)

    # Dự đoán
    predicted_price = model.predict(input_array)[0]

    # Hiển thị kết quả
    st.success(f"💰 Giá nhà dự đoán: **{predicted_price:,.0f} triệu đồng**")
