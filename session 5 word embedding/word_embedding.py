# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:18:19 2024

@author: ylepen
"""

import pandas as pd
import numpy as np

#%%
import os 
os.environ['GENSIM_DATA_DIR']='./models'

#%%
import gensim.downloader as api
info_df=pd.DataFrame.from_dict(api.info()['models'],orient='index')
info_df[['file_size','base_dataset','parameters']].head(5)

#%%
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

#%%
model.most_similar(positive=['woman','computer'],topn=3)
model.most_similar(positive=['man','computer'],topn=3)

model.most_similar(positive=['man','homemaker'],topn=3)
model.most_similar(positive=['woman','homemaker'],topn=3)

