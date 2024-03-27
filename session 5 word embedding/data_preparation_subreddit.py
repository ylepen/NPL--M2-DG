# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:06:37 2024

@author: ylepen
"""

import pandas as pd
import numpy as np


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
import spacy
import textacy

nlp=spacy.load('en_core_web_sm',disable=['parser','ner'])

#%%
df['text']=df['title']+':'+df['text']

#%%
import textacy
import textacy.preprocessing as tprep

def normalize(text):
    text = tprep.replace.urls(text)# we replace url with text
    text = tprep.remove.html_tags(text)
    text = tprep.normalize.hyphenated_words(text)
    text = tprep.normalize.quotation_marks(text)
    text = tprep.normalize.unicode(text)
    text = tprep.remove.accents(text)
    text = tprep.remove.punctuation(text)
    text = tprep.normalize.whitespace(text)
    text = tprep.replace.numbers(text)
    return text

#%%
df['text'].apply(normalize)
#%%
def extract_lemmas(doc,**kwargs):
    textacy.extract.words(doc,
                                   filter_stops=True,
                                   filter_punct=True,
                                   filter_nums=True,
                                   include_pos=['ADJ','NOUN'],
                                   exclude_pos= None,
                                   min_freq=1)
    return[t.lemma_ for t in textacy.extract.words(doc,**kwargs)]
#%%
def extract_nlp(doc):
    return{
        'lemmas' : extract_lemmas(doc,exclude_pos=['PART','PUNCT','DET','PRON','SYM','SPACE'], filter_stops = False)
        }
#%%
df['lemmas']=None
#%%
if spacy.prefer_gpu():
    print("working on GPU.")
else:
    print("No GPU found, working on CPU")
    

#%%
    
batch_size = 50

for i in range(0,len(df),batch_size):
    docs=nlp.pipe(df['text'][i:i+batch_size])
    for j, doc in enumerate(docs):
            for col,values in extract_nlp(doc).items():
                df[col].iloc[i+j]=values
#%%

df['lemmas']=df['lemmas'].apply(lambda items:' '.join(items))

#%%
import sqlite3
db_name = "reddit-selposts.db"
con =sqlite3.connect(db_name)
df.to_sql('posts', con, index=False,if_exists='replace')
con.close()