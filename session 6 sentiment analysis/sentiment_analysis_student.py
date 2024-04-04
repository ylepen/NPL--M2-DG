# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:04:24 2024

@author: ylepen
"""

import pandas as pd
import numpy as np
import nltk


#%%
df = pd.read_json('C:/Users/ylepen/OneDrive - Université Paris-Dauphine/COURS Dauphine/NLP/sentiment analysis/data/reviews_.json',lines=True)

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

#%% Comparison of the different lexicons
