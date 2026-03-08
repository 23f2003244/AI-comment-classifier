import streamlit as st
import pickle 
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("AI Comment Classifier")
text = st.text_area("Enter a comment")

if st.button("Predict"):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    st.success(f"Prediction: {prediction[0]}")