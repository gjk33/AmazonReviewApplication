#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 14:06:57 2019

@author: Greg
"""

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
from nltk.corpus import stopwords
import pickle 
import string
import numpy as np
from nlp_features import remove_punctuations, tokenize, transformSentiment, transformScore
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample  
from sklearn.preprocessing import FunctionTransformer


def setupHelpfulness():
            print("BEGIN LOAD - HELPFULNESS")
            df = pd.read_json('/Users/Greg/amazonreviewsite/app/setup/reviewdata.json')

            df['helpful_numerator'], df['helpful_denominator'] = zip(*df.pop('helpful'))
            df= df[df['helpful_denominator'] >= 5]
            df['helpful_ratio'] = df.helpful_numerator / df.helpful_denominator 
            df["helpful_bool"] = df["helpful_ratio"].map(lambda x: x >= 0.8)
            x = df.helpful_bool.value_counts()
            df_true = df[df.helpful_bool == True]
            df_true.reset_index(drop=True, inplace = True)
            df_false = df[df.helpful_bool == False]

            
            from sklearn.utils import resample  
            df_false_upsampled = resample(df_false, replace = True, n_samples = 5132)
            df_false_upsampled.reset_index(drop=True,inplace=True)


            df = pd.concat([df_true, df_false_upsampled] , ignore_index=True)

            print("Pre Transformation")
            print(x)
            print("After Transformation")
            print(df.helpful_bool.value_counts())
            print("BEGINNING CLEAN")
            df.reviewText = df.reviewText.apply(remove_punctuations)
            stop = stopwords.words('english')
            df.reviewText = df.reviewText.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])) 
            print("END CLEAN")
                
                
            print("BEGINNING TFIDF")


            X_train, X_test, Y_train, Y_test = train_test_split(
                list(df.reviewText), df["helpful_bool"], test_size=0.2)
        
            

            clf = LogisticRegression(solver = 'lbfgs')

                
            Gvectorizer = TfidfVectorizer(use_idf=True, tokenizer = tokenize, token_pattern=u'(?u)\b\w*[a-zA-Z]\w*\b', min_df = 0.0005, max_df = 0.7)



            text_clf = Pipeline([('tfidf' , Gvectorizer) ,('clf', clf)])

            text_clf.fit(X_train,Y_train)
            predicted = text_clf.predict(X_test)
            score = np.mean(predicted == Y_test)
            print(score)
            return text_clf


def setupOverall():
            print("BEGIN LOAD - OVERALL")
            
            df = pd.read_json('reviewdata.json')
            
            df["overall_class"] = df["overall"].apply(transformScore)
            df_high = df[df.overall_class == 'High']
            df_high.reset_index(drop=True, inplace = True)
            df_med = df[df.overall_class == 'Med']
            df_low = df[df.overall_class == 'Low']
            x = df['overall_class'].value_counts().get('High')

            df_med_upsampled = resample(df_med, replace = True, n_samples = x)
            df_med_upsampled.reset_index(drop=True,inplace=True)


            df_low_upsampled = resample(df_low, replace = True, n_samples = x)
            df_low_upsampled.reset_index(drop=True,inplace=True)

            df = pd.concat([df_high, df_med_upsampled, df_low_upsampled], ignore_index=True)

            X_train, X_test, Y_train, Y_test = train_test_split(list(df.reviewText), df["overall_class"], test_size=0.2)
            
                
            clf = LogisticRegression(solver = 'lbfgs')

            sentiment = FunctionTransformer(transformSentiment, validate = False)

                        
            Gvectorizer = TfidfVectorizer(use_idf=True, tokenizer = tokenize, token_pattern=u'(?u)\b\w*[a-zA-Z]\w*\b', min_df = 0.0005, max_df = 0.7)
            from sklearn.pipeline import FeatureUnion


                
            text_clf = Pipeline([
                    ('features', FeatureUnion(
                            [('tfidf' , Pipeline([
                                    ('tfidf', Gvectorizer)
                                    ])),
                            ('sentiment', Pipeline([
                                    ('sentiment', sentiment)
                                    ]))
                            ])),
                    ('clf' , clf)
                    ])
            text_clf.fit(X_train,Y_train)
            predicted = text_clf.predict(X_test)
            score = np.mean(predicted == Y_test)
            print(score)
            return text_clf



    
    

    

    



    
def setupType():
            print("BEGIN LOAD - TYPE ")
            
            df = pd.read_json('reviewdata.json')
            
        
        
            print("BEGINNING CLEAN")
            df.reviewText = df.reviewText.apply(remove_punctuations)
#            token_pattern=r'(?u)\b[A-Za-z]+\b'
            stop = stopwords.words('english')
            df.reviewText = df.reviewText.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])) 
            print("END CLEAN")

        
        
            print("BEGINNING TFIDF")
           
            
            X = list(df["reviewText"])
            Y = list(df["product_type"])
        
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
          
            Gvectorizer = TfidfVectorizer(use_idf=True, tokenizer = tokenize, token_pattern=u'(?u)\b\w*[a-zA-Z]\w*\b',min_df = 0.0005, max_df = 0.7)
                  
            text_clf = Pipeline([('tfidf', Gvectorizer),('clf', LogisticRegression())])
            text_clf.fit(X_train,Y_train)
            predicted = text_clf.predict(X_test)
            score = np.mean(predicted == Y_test)
            print(score)
            return text_clf
            
    
def save(model, filepath):
    with open(filepath, 'wb') as out_file:
        pickle.dump(model, out_file)
    return model


def main():            
    typeClassifier = setupType()
    filename = 'TypeClassifier.pickle'
    save(typeClassifier, filename)
    overallClassifier = setupOverall()
    filename = 'OverallClassifier.pickle'
    save(overallClassifier, filename)
    helpfulnessClassifier = setupHelpfulness()
    filename = 'HelpfulnessClassifier.pickle'
    save(helpfulnessClassifier, filename)
    


   


if __name__ == '__main__':
    main()