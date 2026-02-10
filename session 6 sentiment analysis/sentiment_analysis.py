# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:04:24 2024

@author: ylepen
"""

import pandas as pd
import nltk
import matplotlib.pyplot as plt

#%%
df = pd.read_json('C:/Users/ylepen/OneDrive - Université Paris-Dauphine/COURS Dauphine/NLP/session 7 sentiment analysis/data/reviews_.json',lines=True)

#%%
df.rename(columns={"reviewText":"text"},inplace=True)
df = df[df['text'].str.len()!=0]

#%% selection of text with one sentence only
from nltk.tokenize import sent_tokenize
df['number_of_sentences']=df['text'].apply(lambda text : len(sent_tokenize(text)))

df=df[df['number_of_sentences']==1]

#%%
df.info()
df['text'].sample(5)
#%%#####################################
#Bin-Liu lexicon 
########################################

from nltk.corpus import opinion_lexicon
from nltk.tokenize import word_tokenize
nltk.download('opinion_lexicon')

#%%
print('Total number of words in opinion lexicon',len(opinion_lexicon.words()))
print('Example of positive words in opinion lexicon', opinion_lexicon.positive()[:10])
print('Example of negative words in opinion lexicon', opinion_lexicon.negative()[:10])

#%%
# We define a dictionary of positive and negative word for Bin-Liu Lexicon 
pos_score=1
neg_score=-1
word_dict={}

# adding the positive words to the dictionary
for word in opinion_lexicon.positive():
    word_dict[word]=pos_score


# adding the negative words to the dictionary
for word in opinion_lexicon.negative():
    word_dict[word]=neg_score


#%%
def bing_liu_score(text):
    sentiment_score = 0
    bag_of_words = word_tokenize(text.lower())
    for word in bag_of_words:
        if word in word_dict:
            sentiment_score += word_dict[word]
    return sentiment_score / len(bag_of_words)

#%%
df['Bing_Liu_Score']=df['text'].apply(bing_liu_score)
df[['asin','text','Bing_Liu_Score']].sample(3)

#%%
from sklearn.preprocessing import scale
df['Bing_Liu_Score']=scale(df['Bing_Liu_Score'])

df.groupby('overall').agg({'Bing_Liu_Score':'mean'})

#%%#########################################################################
# Textblob
############################################################################
from textblob import TextBlob
sentence1='The sun is shining and the sky is blue'
blob1=TextBlob(sentence1)
print(blob1.polarity)
print(blob1.subjectivity)
#%%
pol_subj=TextBlob(df.loc[33,'text']).sentiment
print(pol_subj)

#%%
df['textblob_Score']=df['text'].apply(lambda text : TextBlob(text).sentiment.polarity)

#%%
df.groupby('overall').agg({'textblob_Score':'median'})
df.groupby('overall').agg({'textblob_Score':'mean'})


#%%
# AFINN
#########################################################################

afinn_wl_url = ('https://raw.githubusercontent.com'
                '/fnielsen/afinn/master/afinn/data/AFINN-111.txt')
afinn_wl_df = pd.read_csv(afinn_wl_url,
                          header=None, # no column names
                          sep='\t',  # tab sepeated
                          names=['term', 'value']) #new column names
seed = 808 # seed for sample so results are stable
afinn_wl_df.sample(10, random_state = seed)

#%%
# Computing Afinn score

from afinn import Afinn
afn=Afinn(emoticons=True)

df['affin_Score']=df['text'].apply(lambda text : afn.score(text))

df.groupby('overall').agg({'affin_Score':'median'})
df.groupby('overall').agg({'affin_Score':'mean'})
df.groupby('overall').agg({'affin_Score':'min'})

#%%
# VADER lexicon
#############################################################
nltk.downloader.download('vader_lexicon')
#%%
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
Vader=[]
for text in df['text']:
    output_index_dict=analyzer.polarity_scores(text)
    Vader.append(output_index_dict['compound'])
    
df['Vader_Score']=Vader

df.groupby('overall').agg({'Vader_Score':'mean'})
#%%
# sentiwordnet

from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn

nltk.download('sentiwordnet')
nltk.download('wordnet')

sentence = 'the painting is dark and dreadful'

token_sentence = nltk.word_tokenize(sentence)

#%% Comparison of the different lexicons
df['sentiment']=0
df.loc[df['overall']>=3,'sentiment']=1
#df.loc[df['overall']<3,'sentiment']=0

#%%
df['sentiment2']=(df['overall']>=3).astype(int)

#%% Bing Liu score
df['Bing_Liu_Score'].min()
df['Bing_Liu_predicted']=0
df['Bing_Liu_predicted']=df['Bing_Liu_predicted'].mask(df['Bing_Liu_Score']>0,1)

from sklearn.metrics import classification_report
print(classification_report(df['sentiment'], df['Bing_Liu_predicted']))

#%% Textblob score
df['textblob_Score'].min()
df['textblob_predicted']=0
df['textblob_predicted']=df['textblob_predicted'].mask(df['textblob_Score']>0,1)# threshold a 0.1

from sklearn.metrics import classification_report,  ConfusionMatrixDisplay
print(classification_report(df['sentiment'], df['textblob_predicted']))
ConfusionMatrixDisplay.from_predictions(df['sentiment'], df['textblob_predicted'])
plt.show()

#%%
# Vader Score
df['Vader_predicted']=0
df['Vader_predicted']=df['Vader_predicted'].mask(df['Vader_Score']>0,1)# threshold a 0.1

from sklearn.metrics import classification_report
print(classification_report(df['sentiment'], df['Vader_predicted']))

#%%
# affin Score
df['affin_predicted']=0
df['affin_predicted']=df['affin_predicted'].mask(df['affin_Score']>0,1)# threshold a 0.1

from sklearn.metrics import classification_report
print(classification_report(df['sentiment'], df['affin_predicted']))

#%%
# Harvard IV-4 dictionary
import pysentiment2 as ps
hiv4 = ps.HIV4()
text = 'The stock market is booming as the growth expectations are very optimistic'
tokens = hiv4.tokenize(text)
score = hiv4.get_score(tokens)
print(score)

#Loughran and McDonald
lm=ps.LM()
score_lm = lm.get_score(tokens)

#%%

df_tweet = pd.read_csv('C:/Users/ylepen/OneDrive - Université Paris-Dauphine/COURS Dauphine/NLP/session 7 sentiment analysis/data/tweets_remaining_09042020_16072020.csv',sep=';')

df_tweet.info()

df_tweet['token']=df_tweet['full_text'].apply(hiv4.tokenize)

df_tweet['token'].sample(5)

df_tweet['hiv4_score']=df_tweet['token'].apply(hiv4.get_score)

df_tweet['hiv4_score'].sample(10)

#%%
