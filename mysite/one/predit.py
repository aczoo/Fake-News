import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import re,string
import nltk
from string import punctuation
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

def clean_text(text):
    text = re.sub('\[[^]]*\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = " ".join([x for x in text.split() if x.strip().lower() not in stop])
    text = remove_stopwords(text)
    return text
def make_model():
    np.random.seed(42)
    news = pd.read_csv('static/news.csv', header=None)
    news.drop(news.columns[[0]], axis=1, inplace=True)
    columns = news.iloc[0]
    news = news[1:]
    news.columns =columns
    stop = set(stopwords.words('english'))
    punctuation = list(punctuation)
    stop.update(punctuation)
    news['text']=news['text'].apply(clean_text)
    nums = {"REAL":0,"FAKE":1}
    X_train, X_test, y_train, y_test = train_test_split(news["text"],[nums[x] for x in news["label"]], test_size=0.25, random_state=42)
    vectorizer =TfidfVectorizer(stop_words='english', max_df=0.7)
    x_vector_train=vectorizer.fit_transform(X_train)
    x_vector_test = vectorizer.transform(X_test)
    pac=PassiveAggressiveClassifier(max_iter=50)    
    pac.fit(x_vector_train,y_train)