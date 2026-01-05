# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:06:37 2024

@author: ylepen
"""

import pandas as pd
#import numpy as np


#%%
posts_file = 'C:/Users/ylepen/OneDrive - Université Paris-Dauphine/COURS Dauphine/NLP/seance 2/rspct.tsv'
post_df=pd.read_csv(posts_file,sep='\t')

subred_file = "C:/Users/ylepen/OneDrive - Université Paris-Dauphine/COURS Dauphine/NLP/seance 2/subreddit_info.csv"
subred_df = pd.read_csv(subred_file).set_index(['subreddit'])

df=post_df.join(subred_df,on='subreddit')

#%%
column_mapping = {
    'id':'id',
    'subreddit':'subreddit',
    'title':'title',
    'selftext':'text',
    'category_1':'category',
    'category_2':'subcategory',
    'category_3': None,
    'in_data': None,
    'reason_for_exclusion': None
}

columns=[c for c in column_mapping.keys() if column_mapping[c] != None]

df=df[columns].rename(columns=column_mapping)

df=df[df['category']=='autos']


#%%
df['text']=df['title']+':'+df['text']

#%%

import nltk
nltk.download('wordnet')

#%%

from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    #rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_url)  
    #tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)
#%%

df['cleantext']=df['text'].map(lambda s:preprocess(s)) 

#%%
import sqlite3
db_name = "reddit-selposts.db"
con =sqlite3.connect(db_name)
df.to_sql('posts', con, index=False,if_exists='replace')
con.close()