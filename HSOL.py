import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import string
import pickle

# Load NLTK stopwords
nltk.download('stopwords')
stopword = set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")

# Define text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    return " ".join(text)

# Load datasets
hate = pd.read_csv("train.csv")
hate_offensive = pd.read_csv("labeled_data.csv")

# Preprocess datasets
hate.drop('id', axis=1, inplace=True)
hate_offensive.drop(['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither'], axis=1, inplace=True)
hate_offensive["class"].replace({0: 1, 2: 0}, inplace=True)
hate_offensive.rename(columns={'class': 'label'}, inplace=True)

# Combine datasets
frame = [hate, hate_offensive]
df = pd.concat(frame)
df['tweet'] = df['tweet'].apply(clean_text)

# Split data into training and testing sets
x = df['tweet']
y = df['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Vectorize text data using TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)  # Limit to 5000 features
x_train_vectorizer = tfidf.fit_transform(x_train)
x_test_vectorizer = tfidf.transform(x_test)

# Train model
log_reg = LogisticRegression(max_iter=1000)  # Increased the number of iterations
nb = MultinomialNB()
dt = DecisionTreeClassifier()
ensemble = VotingClassifier(estimators=[('lr', log_reg), ('nb', nb), ('dt', dt)], voting='hard')
ensemble.fit(x_train_vectorizer, y_train)

# Save the model and vectorizer
with open('model.pkl', 'wb') as model_file:
    pickle.dump((tfidf, ensemble), model_file)

# Streamlit app
st.title("Hate Speech and Offensive Language Detection")

st.header("Input Comment")
input_tweet = st.text_area("Enter the comment you want to analyze:")

if st.button("Analyze"):
    if input_tweet:
        input_tweet_clean = clean_text(input_tweet)

        try:
            # Load the model and vectorizer
            with open('model.pkl', 'rb') as model_file:
                tfidf_loaded, ensemble_loaded = pickle.load(model_file)

            # Vectorize the input tweet
            input_vectorized = tfidf_loaded.transform([input_tweet_clean])

            # Predict sentiment
            prediction = ensemble_loaded.predict(input_vectorized)[0]
            st.write(f"The sentiment of the tweet is: {'Hate Speech' if prediction == 1 else 'Non-Hate Speech'}")

        except EOFError:
            st.write("Error: Unable to load the model. Please check if the model file exists and is valid.")
    else:
        st.write("Please enter a tweet to analyze.")
