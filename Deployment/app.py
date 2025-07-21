import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

import pandas as pd  

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r'\@w+|\#','', text)
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.Series(X).apply(self.clean_text).tolist()  

pipeline = joblib.load("sentiment_pipeline.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("Sentiment Analysis Twitter")
st.write("Masukkan teks tweet untuk memprediksi sentimennya.")

text_input = st.text_area("Masukkan Teks di Sini", height=150)

if st.button("Prediksi"):
    if text_input.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        pred = pipeline.predict([text_input])[0]
        label = label_encoder.inverse_transform([pred])[0]
        st.success(f"**Hasil Prediksi Sentimen: {label.upper()}**")
