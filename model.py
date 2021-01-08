
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import nltk
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
from sklearn.linear_model import LogisticRegression
import pickle


df = pd.read_csv("cleandata2.csv")
df.reset_index(inplace= True)


# Preparing email texts into word count matrix format 
emails_bow = CountVectorizer().fit(df.CleanContent)

# For all messages
all_emails_matrix = emails_bow.transform(df.CleanContent)

X = all_emails_matrix
Y = df.Class


classifier = LogisticRegression(max_iter=500,random_state=0)
classifier.fit(X,Y)



# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))


#saving transformation matrix
with open('matrix.pkl', 'wb') as f:
    pickle.dump(emails_bow, f)


# Loading model
model = pickle.load(open('model.pkl','rb'))


# loading the matrix
with open('matrix.pkl', 'rb') as f:
    emails_bow1 = pickle.load(f)


