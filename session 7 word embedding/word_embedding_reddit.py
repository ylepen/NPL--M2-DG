# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:54:41 2024

@author: ylepen
"""

import pandas as pd
import numpy as np

#%%
import sqlite3
db_name='reddit-selposts.db'
con=sqlite3.connect(db_name)
df = pd.read_sql_query("select * FROM posts", con)
con.close()
df

#%%

df['text'] = df['cleantext'].str.lower().str.split()

#%%

sents= df['text']

#%%
from gensim.models.phrases import Phrases, npmi_scorer

phrases = Phrases( sents, min_count=10, threshold=0.3, delimiter='_', scoring=npmi_scorer)

#%%
sent="I had to replace the timing belt in my mercedes c300".split()
phrased =phrases[sent]
print('|'.join(phrased))

#%%

phrase_df = pd.DataFrame.from_dict(phrases.export_phrases(),orient='index',columns=['score'])
phrase_df['phrase']=phrase_df.index
phrase_df = phrase_df[['phrase','score']].drop_duplicates().sort_values(by='score',ascending=False).reset_index(drop=True)
phrase_df[phrase_df['phrase'].str.contains('mercedes')]
#%%
phrase_df[phrase_df['phrase'].str.contains('harley')]
phrase_df[phrase_df['phrase'].str.contains('bmw')]
#%%

phrases = Phrases( sents, min_count=10, threshold=0.7, delimiter='_', scoring=npmi_scorer)

df['phrase_lemmas']=df['text'].map(lambda s: phrases[s])
sents= df['phrase_lemmas']

#%%
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

model= Word2Vec(sents,
                vector_size=100,
                window=2,
                sg=1,
                negative=5,
                min_count=5,
                workers=4,
                epochs=5
                )

#%%

model.wv.most_similar(positive=['bmw'],topn=10)

#%%
model.wv.most_similar(positive=['mercedes','yaris'], negative=['toyota'], topn=5)

#%%


model.save('autos_w2v_100_2_full.bin')

#%%
from gensim.models import FastText

model2=FastText(sents,
                vector_size=100,
                window=2,
                sg=1,
                negative=5,
                min_count=5,
                workers=4,
                epochs=5
                )

model2.save('autos_ft_sg_5.bin')

#%%

model2.wv.most_similar(positive=['bmw'],topn=10)

#%%

from umap import UMAP
words=list(model.wv.key_to_index.keys())
wv = [model.wv[word] for word in words]

reducer =UMAP(n_components=2, metric = 'cosine', n_neighbors=15, min_dist=0.1)
reduced_wv=reducer.fit_transform(wv)

#%%
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'


plot_df=pd.DataFrame(reduced_wv, columns=['x','y'])
plot_df['word'] = words
params={'hover_data': {c: False for c in plot_df.columns}, 'hover_name': 'word'}

fig =px.scatter(plot_df,x='x',y='y',opacity=0.3,size_max=3, **params)
fig.show()

#%%

