from flask import render_template, flash
from app import app 
from app.forms import ReviewForm
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import joblib
import string
import numpy as np
from nltk.corpus import stopwords 



type_file_in = open('/Users/Greg/amazonreviewsite/app/setup/TypeClassifier.pickle',"rb") 
typeClassifier =  joblib.load(type_file_in)
overall_file_in = open('/Users/Greg/amazonreviewsite/app/setup/OverallClassifier.pickle',"rb") 
overallClassifier =  joblib.load(overall_file_in)
helpful_file_in = open('/Users/Greg/amazonreviewsite/app/setup/HelpfulnessClassifier.pickle',"rb") 
helpfulClassifier =  joblib.load(helpful_file_in)


def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    lemm = WordNetLemmatizer()
    for item in tokens:
        stems.append(lemm.lemmatize(item))
    return stems

def construct_class_words(pipeline):
    class_words = {}
    vec = pipeline.steps[0][1]
    clf = pipeline.steps[1][1]
    classes = clf.classes_
    feature_names = vec.get_feature_names()
    for i, class_label in enumerate(classes):
        top100 = np.argsort(clf.coef_[i])[-200:]
        top_names = list()
        for j in top100:
            top_names.append(feature_names[j])
        class_words[class_label] = top_names
    return class_words

def topThreePredictions(typeClassifier, x):
    import operator
    predictions = list(zip(typeClassifier.classes_, typeClassifier.predict_proba([x])[0]))
    predictions.sort(key = operator.itemgetter(1),  reverse = True)    
    return predictions[:3]

type_class_words = construct_class_words(typeClassifier)





@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET','POST'])
def index():
        stop = stopwords.words('english')
        form = ReviewForm()
        if form.validate_on_submit():
            review = remove_punctuations(form.reviewText.data)
            review = ' '.join([word for word in review.split() if word not in (stop)])
            if len(review.split()) == 0:
                return render_template('index.html', title = 'Home', form = form, not_enough = True)
            typeClassification = typeClassifier.predict([review])[0]
            overallClassification = overallClassifier.predict([review])[0]
            helpfulClassification = helpfulClassifier.predict([review])[0]
            reviewTokens = tokenize(review)
            type_keywords = ", ".join(list(set(type_class_words.get(typeClassification)).intersection(set(reviewTokens))))

            if form.resultPresentation.data == 'tt':
                topThree = topThreePredictions(typeClassifier, review)
                return render_template('index.html', title = 'Home', form = form, topThree = topThree, overallClassification = overallClassification, helpfulClassification = helpfulClassification , type_keywords = type_keywords)          
            return render_template('index.html', title = 'Home', form = form, typeClassification = typeClassification, overallClassification = overallClassification, helpfulClassification = helpfulClassification, type_keywords = type_keywords)
        return render_template('index.html', title = 'Home', form = form)