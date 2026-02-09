# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:18:19 2024

@author: ylepen
"""

import pandas as pd

#%%
import os 
os.environ['GENSIM_DATA_DIR']='./models'

#%% Import word2vec
# downloard API for gensim: an API for downloading, detting information and loadaing datasets and models
import gensim.downloader as api
print(api.info(name_only='TRUE'))# return dict with info about available corpora(dataset)/models

#%%
print(list(api.info()['corpora'].keys()))
# Show all available models in gensim-data
print(list(api.info()['models'].keys()))
#%%

import json
print(json.dumps(api.info(),indent=2))

#%% displaying the models and their characteristics in a dataframe
info_df=pd.DataFrame.from_dict(api.info()['models'],orient='index')
print(info_df[['base_dataset','parameters']].head(5))

#%%
# we download the pre-trained 'word2vec-google-news-300' model
wv = api.load('word2vec-google-news-300')

#%%
# vocabulary of the model
# wv.index_to_key : size of the vocabulary
print("size of the vocabulary = ",len(wv.index_to_key))

print(wv.index_to_key[:10])

#%%
v_king = wv['king']
print('longueur du vecteur :',len(v_king))
print(v_king)

#%%
# Computation cosine similarity between two words
wv.similarity('king','queen')
wv.similarity('king','man')
wv.similarity('king','car')
#%%
#  Most similar words

wv.most_similar('king',topn=5)

wv.most_similar(positive=['king','queen'],topn=5) # excluds king and queens from the research

wv.most_similar(positive=['king','queen'],negative=['man'],topn=5)
#%%
wv.similar_by_vector(wv['king'],topn=5)
#%%
wv.doesnt_match(['house','car','vehicle','motocar'])

#%%
print(wv.most_similar(positive=['man','homemaker'],topn=5))
print(wv.most_similar(positive=['woman','homemaker'],topn=5))

#%%
# Training a specific model
# Each input to the model must be a list of phrases
from gensim.models import Word2Vec
api.info('fake-news')
dataset = api.load('fake-news')
model=Word2Vec(dataset)
print(len(model.wv.index_to_key))
print(model.wv.index_to_key)
model.wv['comments']
#%% Glove 
model=api.load('glove-wiki-gigaword-50')
#%%
v_king = model['king']
v_queen =model['queen']

print("vector size:", model.vector_size)
print("v_king", v_king[:10])
print("v_queen", v_queen[:10])
print("similarity:", model.similarity('king','queen'))

#%%
model.most_similar('king',topn=3)
model.most_similar('queen',topn=3)

#%%
v_lion = model['lion']
v_nano = model['nanotechnology']

model.cosine_similarities(v_king,[v_queen,v_lion,v_nano])

# Note that word embbeding can have negative values
# similarity values in [-1,+1]

#%% Utilization of the function most_similar

model.most_similar(positive=['woman','king'],negative=['man'],topn=3)

model.most_similar(positive=['paris','germany'],negative=['france'],topn=3)

model.most_similar(positive=['france','berlin'],topn=3)

model.most_similar(positive=['greece','capital'],topn=3)

#%% Biaises in the training set
model.most_similar(positive=['woman','computer'],topn=3)
model.most_similar(positive=['man','computer'],topn=3)

model.most_similar(positive=['man','homemaker'],topn=3)
model.most_similar(positive=['woman','homemaker'],topn=3)

