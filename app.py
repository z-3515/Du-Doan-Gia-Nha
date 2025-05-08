import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model/linear_model.pkl")

st.set_page_config(page_title="Dự đoán giá nhà", layout="centered")

st.title("🏠 Dự đoán giá nhà tại TP. Hồ Chí Minh")
st.markdown("Nhập thông tin bên dưới để dự đoán tổng giá trị căn nhà (đơn vị: triệu đồng).")

# Form người dùng nhập thông tin
with st.form("form_du_doan"):
    dien_tich = st.number_input("Diện tích (m²)", min_value=10.0, max_value=1000.0, step=1.0)
    gia_m2 = st.number_input("Giá theo m² (triệu/m²)", min_value=1.0, max_value=1000.0, step=1.0)
    so_phong_ngu = st.number_input("Số phòng ngủ", min_value=0, max_value=10, step=1)
    so_phong_tam = st.number_input("Số phòng tắm", min_value=0, max_value=10, step=1)
    submit = st.form_submit_button("🔍 Dự đoán")

# Dự đoán khi người dùng nhấn nút
if submit:
    input_data = pd.DataFrame([{
        "dien_tich": dien_tich,
        "gia_m2": gia_m2,
        "so_phong_ngu": so_phong_ngu,
        "so_phong_tam": so_phong_tam
    }])
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Giá nhà dự đoán: **{prediction:,.0f} triệu đồng**")
