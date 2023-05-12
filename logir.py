# Import libraries
import pandas as pd
import numpy as np
import re
import nltk
import pickle
import pandas as pd
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

# Create the logistic regression model
model = LogisticRegression(C=10, max_iter=1000)
model.fit(X_train_tfidf, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

