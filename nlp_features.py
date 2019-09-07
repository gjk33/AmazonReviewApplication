
from nltk.stem import WordNetLemmatizer
import nltk
import string


   


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

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
def analyseSentiment(x):
#        #print(textToBeAnalysed)
#        tb = TextBlob(textToBeAnalysed)
  
        x = sid.polarity_scores(x)["compound"]
        #ss = sid.polarity_scores(textToBeAnalysed)
        #return ss.get('compound')
        
        return (x + 1 ) / 2
   
def transformSentiment(x):
    for i in x :
        i = analyseSentiment(i)

def transformScore(x):
    if x > 3:
        return "High"
    elif x < 3:
        return "Low"
    else:
        return "Med"    
