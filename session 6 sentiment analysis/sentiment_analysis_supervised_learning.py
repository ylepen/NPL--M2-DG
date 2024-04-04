# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 18:52:32 2024

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

#%% Cleaning the text
nltk.download('wordnet')

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
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    #tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    #stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words)

#%%

df['text']=df['text'].map(lambda s:preprocess(s)) 

#%%
# Vizualisation 
from wordcloud import WordCloud
positive_comment=df[df['sentiment']==1]['text']
negative_comment=df[df['sentiment']==0]['text']

# Sample some positive and negative tweets to create word clouds
sample_positive_text = " ".join(text for text in positive_comment.sample(frac=0.1, random_state=23))
sample_negative_text = " ".join(text for text in negative_comment.sample(frac=0.1, random_state=23))

wordcloud_positive = WordCloud(width=800,height=400, max_words=200,background_color="white").generate(sample_positive_text)
wordcloud_negative = WordCloud(width=800, height=400, max_words=200, background_color="white").generate(sample_negative_text)

# Display the generated image using matplotlib
plt.figure(figsize=(15, 7.5))

# Positive word cloud
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.title('Favorable text Word Cloud')
plt.axis("off")

# Negative word cloud
plt.subplot(1, 2, 2)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.title('Unfavorable text Word Cloud')
plt.axis("off")

plt.show()

#%%
# Step 2: Train- Test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df['text'],df['sentiment'], test_size=0.2, random_state =42, stratify=df['sentiment'])

print('Size of the Training Data',X_train.shape[0])
print('Size of the Test Data',X_test.shape[0])

print('Distribution of the classes in the Training Data :')
print('Positive Sentiment', str(sum(Y_train==1)/len(Y_train)*100.0))
print('Negative Sentiment', str(sum(Y_train==0)/len(Y_train)*100.0))

print('Distribution of the classes in the Test Data :')
print('Positive Sentiment', str(sum(Y_test==1)/len(Y_test)*100.0))
print('Negative Sentiment', str(sum(Y_test==0)/len(Y_test)*100.0))

#%%
# Step 3: Text Vectorization
############################################################

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=10,ngram_range=(1,1))
X_train_tf=tfidf.fit_transform(X_train)
X_test_tf=tfidf.transform(X_test)

##########################################################
#%% Step 4: Training the ML Model
##########################################################

from sklearn.svm import LinearSVC
model1=LinearSVC(random_state=42,tol=1e-5)
model1.fit(X_train_tf,Y_train)
#%%
import joblib
filename = 'text_sentiment_svc.sav'
joblib.dump(model1,filename)
#%%
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

Y_pred=model1.predict(X_test_tf)

print('Accuracy score =',accuracy_score(Y_test, Y_pred))
print('ROC-AUC score =',roc_auc_score(Y_test, Y_pred))

#%%
sample_reviews = df.sample(5)
sample_reviews_tf = tfidf.transform(sample_reviews['text'])
sentiment_prediction = model1.predict(sample_reviews_tf)
sentiment_prediction_df = pd.DataFrame(data = sentiment_prediction,index=sample_reviews.index,columns=['sentiment_prediction'])
sample_reviews=pd.concat([sample_reviews,sentiment_prediction_df],axis=1)
print('Sample review with their sentiment')
sample_reviews[['text_orig','sentiment_prediction']]