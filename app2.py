import streamlit as st
import joblib
import numpy as np

# โหลดโมเดลและ StandardScaler
model = joblib.load("obesity_weight_predictor.pkl")
scaler = joblib.load("scaler.pkl")

def show_app2():
    # UI ของ Streamlit
    st.title("🎯 Obesity-Based Weight Predictor")
    st.write("ใส่ข้อมูลเพื่อทำนาย **น้ำหนัก (Weight)**")

    # เลือก Gender (0 = Female, 1 = Male)
    gender = st.selectbox("Gender", ["Female", "Male"])
    gender_value = 1 if gender == "Male" else 2

    # กรอกส่วนสูง (Height in meters)
    height = st.number_input("Height (meters)", min_value=1.0, max_value=2.5, step=0.01)

    # เลือกระดับ Obesity (ที่เคยแปลงเป็นตัวเลขแล้ว)
    obesity_levels = {
        "Insufficient Weight": 0,
        "Normal Weight": 1,
        "Overweight Level I": 2,
        "Overweight Level II": 3,
        "Obesity Type I": 4,
        "Obesity Type II": 5,
        "Obesity Type III": 6
    }
    obesity = st.selectbox("Obesity Level", list(obesity_levels.keys()))
    obesity_value = obesity_levels[obesity]

    # กดปุ่มทำนาย
    if st.button("Predict Weight"):
        # เตรียมข้อมูลนำเข้าโมเดล
        input_data = np.array([[gender_value, height, obesity_value]])
        input_scaled = scaler.transform(input_data)

        # ทำนายน้ำหนัก
        predicted_weight = model.predict(input_scaled)

        # แสดงผลลัพธ์
        st.success(f"📌 Predicted Weight: {predicted_weight[0]:.2f} kg")
