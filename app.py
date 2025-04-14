import streamlit as st  
import joblib  

# Function to load model and vectorizer with error handling  
def load_model_and_vectorizer():  
    try:  
        model = joblib.load("logistic_regression_best_model.pkl")  
        vectorizer = joblib.load("X_bow_vectorizer.pkl")  
        
        # Check if vectorizer has the transform method  
        if not hasattr(vectorizer, 'transform'):  
            raise ValueError("Vectorizer tidak valid. Silakan periksa file X_bow_vectorizer.pkl.")  
        
        return model, vectorizer  
    except Exception as e:  
        st.error(f"Error saat memuat model atau vectorizer: {str(e)}")  
        return None, None  

# Load model & vectorizer  
model, vectorizer = load_model_and_vectorizer()  

if model is not None and vectorizer is not None:  # Proceed only if loading is successful  
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
