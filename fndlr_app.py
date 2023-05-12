import streamlit as st
import pickle
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from logir import X_train

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a function to preprocess the news headline
def preprocess(text):
  # Convert to lowercase
  text = text.lower()
  # Remove numbers and punctuation
  text = re.sub(r'[^\w\s]|[\d]', '', text)
  # Tokenize the text
  tokens = nltk.word_tokenize(text)
  # Remove stop words
  stop_words = set(stopwords.words('english'))
  tokens = [token for token in tokens if token not in stop_words]
  # Join the tokens back into a string
  text = ' '.join(tokens)
  return text

# Create a function to predict whether the news headline is real or fake
def predict(headline):
  # Preprocess the headline
  headline = preprocess(headline)
  # Vectorize the headline
  vectorizer = TfidfVectorizer()
  vectorizer.fit(X_train)
  headline_vec = vectorizer.transform([headline])
  # Make a prediction
  prediction = model.predict(headline_vec)[0]
  return prediction

# Create a title for the page
st.title('Fake News Detection')

# Create a text input field for the news headline
headline = st.text_input('Enter a news headline:')

# If the user enters a headline, make a prediction and display it
if headline:
  prediction = predict(headline)
  if prediction == 0:
    st.write('The news headline is real.')
  else:
    st.write('The news headline is fake.')
