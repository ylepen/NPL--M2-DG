# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 18:49:33 2026

@author: ylepen
"""

import pandas as pd



from urllib import request
# urllib.request : python library for opening library 


# get the playlist dataset file

data = request.urlopen("https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt")

# request.urlopen(url) : open url 

# we load data from the url 

#1 data.read() read the content of the http 
# return the raw data as bytes code (binary format)

#2 decode('utf-8')
# converts the bytes into a readable string
# uses the utf-8 encoding (most common text encoding)

#3 split('\n')[2,:]
# split the string into a list of lines
# uses the newline character '\n' as a separator
# each line a separate element in the list

# 4. [2,:]
# We remove the first two lines (headers)

lines=data.read().decode("utf-8").split('\n')[2:]

# lines is a list of strings 

# s.string: divide a string into its components 

playlists=[s.rstrip().split() for s in lines if len(s.split())>1]

# load song metadata
songs_file =request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt')
songs_file = songs_file.read().decode("utf-8").split('\n')
print(songs_file[0])
songs = [s.rstrip().split('\t') for s in songs_file]
# the string are split according to tabular \t

songs_df=pd.DataFrame(data=songs,columns=['id','title','artist'])
songs_df =songs_df.set_index('id')


print(playlists[0])

#%%

import gensim.downloader as api
wv = api.load('word2vec-google-news-300')

from gensim.models import Word2Vec

model=Word2Vec(playlists,vector_size=32,window=20,negative=50,min_count=1,workers=4)

song_id = 2172

model.wv.most_similar(positive=str(song_id),topn=5)

print(songs_df.iloc[song_id])

#%%
import numpy as np

def print_recommendations(song_id):
    similar_songs = np.array(
        model.wv.most_similar(positive=str(song_id),topn=5)
        )[:,0]
    return songs_df.iloc[similar_songs]

#%%
print_recommendations(2172)
print_recommendations(3822)
