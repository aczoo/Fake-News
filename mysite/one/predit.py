import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import re,string
import nltk
import string 
import pickle

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
nltk.download('stopwords')
class myModel:
    def __init__ (self):
        self.ourModel = None
        self.stop = None
        self.vectorizer = None
    def clean_text(self,text):
        text = re.sub('\[[^]]*\]', '', text)
        text = re.sub(r'http\S+', '', text)
        text = " ".join([x for x in text.split() if x.strip().lower() not in self.stop])
        return text
    def model(self):
        np.random.seed(42)
        news = pd.read_csv('./one/static/news.csv', header=None)
        news.drop(news.columns[[0]], axis=1, inplace=True)
        columns = news.iloc[0]
        news = news[1:]
        news.columns =columns
        self.stop = set(stopwords.words('english'))
        punctuation = list(string.punctuation)
        self.stop.update(punctuation)
        news['text']=news['text'].apply(self.clean_text)
        nums = {"REAL":0,"FAKE":1}
        X_train, X_test, y_train, y_test = train_test_split(news["text"],[nums[x] for x in news["label"]], test_size=0.25, random_state=42)
        self.vectorizer =TfidfVectorizer(stop_words='english')
        x_vector_train=self.vectorizer.fit_transform(X_train)
        pac=PassiveAggressiveClassifier(max_iter=50)
        pac.fit(x_vector_train,y_train)
        self.ourModel = pac
        filename = 'model.sav'
        pickle.dump(pac, open('model.sav', 'wb'))
        pickle.dump(self.vectorizer, open('vectorizer.sav', 'wb'))

    def predict(self, text):
        vector = self.vectorizer.transform(text)
        value = self.ourModel.predict(vector)
        if value == 1:
            return "FAKE"
        else:
            return "REAL"
if __name__ == '__main__':
   tester = myModel()
   tester.model()
   print(tester.predict(["LOS ANGELES—Acting swiftly to ensure that the necessary demand was fully met, the City of L.A. booked 5,000 hotel rooms Thursday for police officers to take naps in between displacing homeless Angelenos. “LAPD officials need a place to rest and recuperate after long hours of putting up fences, throwing away personal possessions, and dragging individuals from their residences,” said Mayor Eric Garcetti, confirming that the rooms would be located near major homeless encampments so officers could conveniently return to relax after forcing hundreds of unhoused people to disperse at gunpoint. “We anticipate that this may be more hotel rooms than we will actually need, but the spares will allow exhausted officers to really stretch out and relax, and we’ve rented an additional suite of rooms for the LAPD to house their weapons and crowd-control measures. This is only a first step, of course, and our eventual hope is to get these officers into permanent second homes where they can throw things at the unhoused from their front lawn.” Garcetii closed his remarks by urging Los Angeles residents to donate any spare canned goods to help keep up officers’ strength while they’re beating people with batons."]))
