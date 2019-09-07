#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:42:43 2019

@author: Greg
"""

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk
import random 
from nltk.corpus import stopwords
import pickle 
import string
   

print("we starting")


class pickleDatBitch:
    
    
    def remove_punctuations(self,text):
        for punctuation in string.punctuation:
            text = text.replace(punctuation, '')
        return text
    
    def tokenize(self,text):
        tokens = nltk.word_tokenize(text)
        stems = []
        for item in tokens:
            stems.append(PorterStemmer().stem(item))
        return stems
    
    def constructData(self,data, product_type):
        l = list()
        data = data.to_dict(orient='records')
        length = len(data)
        for i in range(length):
            x = (data[i], product_type[i])
            l.append(x)
        return l

    
    def run(self):
            print("BEGIN LOAD")
            df_vg = pd.read_json('Video_Games_5.json',lines = True)
            df_vg = df_vg.sample(5000)
            df_vg["product_type"] = "Video Game"
            df_thi = pd.read_json('Tools_and_Home_Improvement_5.json',lines = True)
            df_thi = df_thi.sample(5000)
            df_thi["product_type"] = "Tools and Home Improvement"
            df_beauty = pd.read_json('Beauty_5.json',lines = True)
            df_beauty = df_beauty.sample(5000)
            df_beauty["product_type"] = "Beauty"
            df_cells = pd.read_json('Cell_Phones_and_Accessories_5.json',lines = True)
            df_cells = df_cells.sample(5000)
            df_cells["product_type"] = "Cell Phones and Accessories"
            df_health = pd.read_json('Health_and_Personal_Care_5.json',lines = True)
            df_health = df_health.sample(5000)
            df_health["product_type"] = "Health"
            df_clothing = pd.read_json('Clothing_Shoes_and_Jewelry_5.json',lines = True)
            df_clothing = df_clothing.sample(5000)
            df_clothing["product_type"] = "Clothing"
        
        
            print("END LOAD")
        
        
            df = pd.concat([df_vg,df_thi,df_beauty, df_cells])
        
            
        
        
            print("BEGINNING CLEAN")
            df.reviewText = df.reviewText.apply(self.remove_punctuations)
#            token_pattern=r'(?u)\b[A-Za-z]+\b'
            stop = stopwords.words('english')
            df.reviewText = df.reviewText.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])) 
            print("END CLEAN")

        
        
            print("BEGINNING TFIDF")
            texts = df.reviewText
            
            vectorizer = TfidfVectorizer(use_idf=True, tokenizer = self.tokenize, token_pattern=u'(?u)\b\w*[a-zA-Z]\w*\b', min_df = 0.01, max_df = 0.5)
            X = vectorizer.fit_transform(texts)
            
            filename = 'vectorizer.pickle'
            pickle.dump(vectorizer, open(filename,"wb"))
#            feature_names = vectorizer.get_feature_names()
        
            idf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
        
            print("END TFIDF")
        
        
            product_type = list(df["product_type"])
    
            
            print("BEGINNING DATA CONSTRUCTION")
            
            
            nltkData = self.constructData(idf_df, product_type)
            random.shuffle(nltkData)
            splitValue = (int)((len(nltkData) - 1) * 0.8)
            training_set, testing_set = nltkData[splitValue:], nltkData[:splitValue]
            print("END DATA CONSTRUCTION")
            
            print("BEGIN CLASSIFICATION")
            #classifier = nltk.NaiveBayesClassifier.train(training_set)
            #print("Naive Bayes accuracy percent:",nltk.classify.accuracy(classifier, testing_set))
            #print(classifier.show_most_informative_features(5))
            
            
            
            from nltk.classify.scikitlearn import SklearnClassifier
            from sklearn.naive_bayes import MultinomialNB
            
            MNB_classifier = SklearnClassifier(MultinomialNB())
            MNB_classifier.train(training_set)
            print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, testing_set))
            
            filename = 'MNB_classifier.sav'
            pickle.dump(MNB_classifier, open(filename,"wb"))
            
            #
            #BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
            #BernoulliNB_classifier.train(training_set)
            #print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)))
            
            
            from sklearn.linear_model import LogisticRegression
            
            LogisticRegression_classifier = SklearnClassifier(LogisticRegression(solver='lbfgs', multi_class = 'multinomial' ))
            LogisticRegression_classifier.train(training_set)
            print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)))
            
            filename = 'LogReg_classifier.sav'
            pickle.dump(LogisticRegression_classifier, open(filename,"wb"))
