import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
import os

st.set_page_config(
    page_title="تشخیص سرطان سینه",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 تشخیص سرطان سینه با یادگیری ماشین")
st.write("مدل نهایی بعد از مقایسه و Tuning روی دیتاست Wisconsin")

MODEL_PATH = os.path.join("models", "best_model.joblib")
SCALER_PATH = os.path.join("models", "scaler.joblib")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        st.error(
            "فایل مدل پیدا نشد.\n"
            "اول `compare_models.py` یا `tune_model.py` را اجرا کنید."
        )
        st.stop()
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model()
data = load_breast_cancer()
feature_names = data.feature_names
defaults = data.data.mean(axis=0)

st.success("مدل نهایی با موفقیت بارگذاری شد")

st.subheader("ویژگی‌های تومور را وارد کنید")
st.caption("مقادیر پیش‌فرض = میانگین دیتاست (برای تست سریع)")

cols = st.columns(2)
user_input = []

for i, name in enumerate(feature_names):
    with cols[i % 2]:
        val = st.number_input(
            label=name,
            value=float(round(defaults[i], 4)),
            format="%.4f",
            key=name
        )
        user_input.append(val)

if st.button("🔍 تشخیص بده", use_container_width=True):
    features = np.array(user_input, dtype=float).reshape(1, -1)
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    st.markdown("---")

    if prediction == 0:
        st.error("**نتیجه: بدخیم (Malignant)**")
    else:
        st.success("**نتیجه: خوش‌خیم (Benign)**")

    st.write(f"احتمال بدخیم: **{probability[0]*100:.1f}%**")
    st.write(f"احتمال خوش‌خیم: **{probability[1]*100:.1f}%**")
    st.progress(float(max(0.0, min(1.0, probability[1]))))

st.markdown("---")
st.caption("این ابزار صرفاً آموزشی است و جایگزین تشخیص پزشک نیست.")