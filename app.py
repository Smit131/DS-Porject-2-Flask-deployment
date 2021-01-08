from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from textblob import TextBlob
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
    fpd = pd.DataFrame(str_features,columns = ['CleanContent'])

    # For input message
    final_features = emails_bow1.transform(fpd.CleanContent)
   
    prediction = model.predict(final_features)
    output = prediction[0]

    return render_template('index.html', prediction_text='Content of mail is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
