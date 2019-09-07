from flask import render_template, flash
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import joblib
import string
import numpy as np
from nltk.corpus import stopwords 


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
        top100 = np.argsort(clf.coef_[i])[::-1][:100]
        top_names = list()
        for j in top100:
            top_names.append(feature_names[j])
        class_words[class_label] = top_names
    return class_words

file_in = open('/Users/Greg/amazonreviewsite/app/setup/classifier.pickle',"rb") 
classifier =  joblib.load(file_in)

class_words = construct_class_words(classifier)

review = "I love how this toothbrush feels on my teeth, my dentist recommended it!"
review2 = "I can't believe how much I don't like this game. I have bought every FIFA since 05 and enjoyed them all. This one however is no different to FIFA 18. Career mode is still the same and the Journey is long, boring and tedious. They clearly only care about FUT which again is no different you just have to start from scratch. The game modes are fun for a couple of games but then forgot about and there is no real reason for the champions League mode. In short FIFA 19 is FIFA 18 with more loading screens and slightly overly complicated menus. Spend a couple of hours transferring players in the settings and you save yourself £50. Go buy Spiderman."
text = classifier.predict([review2])[0] 
print(*class_words.get("Videogames"), sep = "\n")

print(review)
print(text)