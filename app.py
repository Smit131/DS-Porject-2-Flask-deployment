from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
nltk.data.path.append('./nltk_data/')
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

import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

with open('matrix.pkl', 'rb') as f:
    emails_bow1 = pickle.load(f)

    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    str_features = request.form.values()
    fpd = pd.DataFrame(str_features,columns = ['Content'])
  
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer() 

    def preprocess(sentence):
      sentence=str(sentence)
      sentence = sentence.lower()
      sentence=sentence.replace('{html}',"") 
      cleanr = re.compile('<.*?>')
      cleantext = re.sub(cleanr, '', sentence)
      rem_url=re.sub(r'http\S+', '',cleantext)
      rem_num = re.sub('[0-9]+', '', rem_url)
      tokenizer = RegexpTokenizer(r'\w+')
      tokens = tokenizer.tokenize(rem_num)  
      filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
      stem_words=[stemmer.stem(w) for w in filtered_words]
      lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
      return " ".join(filtered_words)

    fpd['CleanContent']=fpd['Content'].map(lambda s:preprocess(s)) 
    cc = fpd.CleanContent[0]
    # For input message
    final_features = emails_bow1.transform(fpd.CleanContent)
   
    prediction = model.predict(final_features)
    output = prediction[0]

    return render_template('index.html', prediction_text='Content of mail is {}'.format(output),clean_text='Clean mail  {}'.format(cc))


if __name__ == "__main__":
    app.run(debug=True)
