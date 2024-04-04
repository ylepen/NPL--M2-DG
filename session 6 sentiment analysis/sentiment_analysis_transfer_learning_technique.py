# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:06:27 2024

@author: ylepen
"""

import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt

#%%
df = pd.read_json('C:/Users/ylepen/OneDrive - Université Paris-Dauphine/COURS Dauphine/NLP/sentiment analysis/data/reviews_.json',lines=True)

# note that reviewerName has missing data
#%% ###########################################
# Step 1: Data preparation
###########################################
#%% Removing text with 0 comments
df.rename(columns={"reviewText":"text"},inplace=True)
df = df[df['text'].apply(len)!=0]

#%% selection of text with one sentence only
from nltk.tokenize import sent_tokenize
df['number_of_sentences']=df['text'].apply(lambda text : len(sent_tokenize(text)))

df=df[df['number_of_sentences']==1]
#%%
df["text_orig"]=df['text'].copy()

#%% Assigning a label based on the overall product rating

df['sentiment']=0
df.loc[df['overall']>=3,'sentiment']=1
df.loc[df['overall']<3,'sentiment']=0
#%%
# Removing unnecessary columns 
df.drop(columns=['reviewTime','unixReviewTime','overall','reviewerID','reviewerName','summary'],inplace=True)
df.sample(3)
df.info()
#%%
# pip install transformers
# pip install ruff==0.1.5
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification

config=BertConfig.from_pretrained('bert-base-uncased', finetuning_tasks='binary')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
