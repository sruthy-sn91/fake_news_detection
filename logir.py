# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('train_data.csv')
df.drop(columns=['id','text'],inplace = True)
df = df.dropna()
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
df['title'] = df['title'].apply(preprocess)
X = df['title'] 
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
def tfidf_vectorizer(X_train, X_test):
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec

X_train_tfidf, X_test_tfidf = tfidf_vectorizer(X_train, X_test)

lr = LogisticRegression(max_iter=1000)

# A parameter grid for the C hyperparameter is created, which is the inverse of the regularization strength. 
# Three different values of C are specified to be tested: 0.1, 1, and 10.
param_grid_lr = {'C': [10]}

# GridSearchCV is used to search for the best combination of hyperparameters using 5-fold cross-validation (cv=5) on the training set
grid_lr = GridSearchCV(lr, param_grid_lr, cv=5)
grid_lr.fit(X_train_tfidf, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(grid_lr, f)

def classify_news(text):
    vectorizer = TfidfVectorizer()
    input_vec = vectorizer.transform([text])
    prediction = model.predict(input_vec)[0]
    return prediction

def main():
    st.title("Fake News Detection")

    user_input = st.text_area("Enter the news headline:", height=100)
    if st.button("Classify"):
        result = classify_news(user_input)
        if result == 0:
            st.write("The news headline is classified as: **Fake**")
        else:
            st.write("The news headline is classified as: **Real**")
        
if __name__ == "__main__":
    # Load the trained model and vectorizer
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    main()
