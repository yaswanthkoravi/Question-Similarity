from flask import Flask, jsonify, request,render_template
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as numpy
import pandas as pd
from bs4 import BeautifulSoup
import distance
import string
from fuzzywuzzy import fuzz
import scipy  as sp
from sklearn.linear_model import LogisticRegression
import joblib
import re
import nltk




app = Flask(__name__)


###################################################


def cleaner(x):  
    x=x.translate(str.maketrans('', '', string.punctuation))
    x=x.lower()
    x=BeautifulSoup(x)
    x=x.get_text()
    return x
def get_longest_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)
###################################################


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    clf = joblib.load('model.pkl')
    tf_vect = joblib.load('tfidf_text.pkl')
    scale = joblib.load('scaler.pkl')
    to_predict_list = request.form.to_dict()
    q1_clean=cleaner(to_predict_list['q1'])
    q2_clean=cleaner(to_predict_list['q2'])
    
    w1 = set(map(lambda word: word.lower().strip(), q1_clean.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2_clean.split(" ")))    
    word_share=1.0 * len(w1 & w2)/(len(w1) + len(w2))


    token_set_ratio= fuzz.token_set_ratio(q1_clean, q2_clean)
    token_sort_ratio= fuzz.token_sort_ratio(q1_clean, q2_clean)
    token_ratio= fuzz.QRatio(q1_clean, q2_clean)
    token_partial_ratio= fuzz.partial_ratio(q1_clean, q2_clean)
    longest_substr_ratio= get_longest_substr_ratio(q1_clean, q2_clean)

    fx=scale.transform([[word_share,token_set_ratio,token_sort_ratio,token_ratio,token_partial_ratio,longest_substr_ratio]])
    q1t=tf_vect.transform([q1_clean])
    q2t=tf_vect.transform([q2_clean])

    X = sp.sparse.hstack([q1t,q2t,fx])

    pred = clf.predict(X)
    if pred[0]:
        prediction = "SIMILAR"
    else:
        prediction = "NOT SIMILAR"

    return render_template('result.html',prediction=prediction,q1=to_predict_list['q1'],q2=to_predict_list['q2'])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
