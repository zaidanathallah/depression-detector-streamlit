import streamlit as st
import joblib

# Load model & vectorizer
model = joblib.load("logistic_regression_best_model.pkl")
vectorizer = joblib.load("X_bow_vectorizer.pkl")

st.set_page_config(page_title="Depression Detector", layout="centered")
st.title("🧠 Depression Detector")
st.markdown("Masukkan teks dari postingan atau tulisan untuk diprediksi apakah mengandung tanda-tanda depresi.")

user_input = st.text_area("Masukkan teks di sini:")

if st.button("Prediksi"):
    if user_input.strip():
        vectorized = vectorizer.transform([user_input])
        prediction = model.predict(vectorized)[0]
        label = "🟡 Depressed" if prediction == 1 else "🟢 Not Depressed"
        st.markdown(f"### Hasil Prediksi: {label}")
    else:
        st.warning("Tolong masukkan teks terlebih dahulu.")