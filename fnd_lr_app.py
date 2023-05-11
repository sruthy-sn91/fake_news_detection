import streamlit as st
import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

nltk.download('punkt')
nltk.download('stopwords')

def preprocess(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove numbers and punctuation
        text = re.sub(r'[^\w\s]|[\d]', '', text)
        text = re.sub(r'\d+', '', text)
        # Tokenize the text
        tokens = nltk.word_tokenize(text)
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        # Join the tokens back into a string
        text = ' '.join(tokens)
        return text

def tfidf_vectorizer(text):
    vectorizer = TfidfVectorizer()
    preprocessed_text = preprocess([text])
    text_vec = vectorizer.transform([preprocessed_text])
    return text_vec

def classify_news(text):
    try:
        input_vec = tfidf_vectorizer([text])
        prediction = model.predict(input_vec)[0]
        return prediction
    except NotFittedError:
        return "The model is not trained yet."


def main():
    st.title("Fake News Detection")

    user_input = st.text_area("Enter the news headline:", height=100)
    if st.button("Classify"):
        result = classify_news(user_input)
        st.write("Prediction:", result)

if __name__ == "__main__":
    # Load the trained model and vectorizer
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
main()
