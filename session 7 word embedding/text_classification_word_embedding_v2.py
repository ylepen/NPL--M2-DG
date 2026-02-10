# -*- coding: utf-8 -*-
"""
Text Classification with Word2Vec Embeddings and XGBoost
Simplified version - classifies emails as spam or not spam
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from gensim.models import Word2Vec
from xgboost import XGBClassifier

# Configuration
DATA_PATH = 'spam.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 42


#%%
#def preprocess_text(text):
#   """Convert text to lowercase and extract words."""
#    text = text.lower()
#  text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
#    words = text.split()
#    return words

#%%
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)  # Better tokenization
    return words
#%%
def get_sentence_embedding(text, model):
    """
    Get sentence embedding by summing word vectors.
    Returns zero vector if no words are in vocabulary.
    """
    words = preprocess_text(text)
    valid_words = [word for word in words if word in model.wv]
    
    if len(valid_words) == 0:
        return np.zeros(model.vector_size)
    
    # average of word vectors embeddings (as specified in requirements)
    return np.mean([model.wv[word] for word in valid_words], axis=0)
#%%

def convert_to_embeddings(texts, model):
    """Convert a series of texts to embedding vectors."""
    return np.array([get_sentence_embedding(text, model) for text in texts])
    #%%

df = pd.read_csv(DATA_PATH, encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
    
print(f"Loaded {len(df)} messages")
print(f"Label distribution:\n{df['label'].value_counts()}\n")
    
    # Split data
X_train, X_test, y_train, y_test = train_test_split(
        df['message'], 
        df['label'],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['label']
    )
    
#%%
# Preprocess training texts for Word2Vec
print("Preprocessing texts...")
sentences = [preprocess_text(text) for text in X_train]
sentences = [s for s in sentences if len(s) > 0]  # Remove empty sentences
print(f"Processed {len(sentences)} sentences\n")
#%%   
EMBEDDING_SIZE = 100 # size of the embedding vector
WINDOW_SIZE = 2 # or 3 maximum distance between the current and the predicted word within a sentence. Should not be too long for emails. 
MIN_COUNT = 2 # ingnore all words with total frequency lower than min_count
EPOCHS = 10# number of iterations over the corpus
sg=1# 1 for skip_gram, 0 for CBOW

# Train Word2Vec model
print("Training Word2Vec model...")
w2v_model = Word2Vec(
        sentences=sentences,
        vector_size=EMBEDDING_SIZE,
        window=WINDOW_SIZE,
        min_count=MIN_COUNT,
        workers=4,
        sg=1,  # Skip-gram
        epochs=EPOCHS
    )
print(f"Vocabulary size: {len(w2v_model.wv)}\n")
#%%  
# Convert texts to embeddings
print("Converting texts to embeddings...")
X_train_embeddings = convert_to_embeddings(X_train, w2v_model)
X_test_embeddings = convert_to_embeddings(X_test, w2v_model)
print(f"Training embeddings shape: {X_train_embeddings.shape}")
print(f"Test embeddings shape: {X_test_embeddings.shape}\n")
#%%   
# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
    
# Train XGBoost classifier
print("Training XGBoost classifier...")
clf = XGBClassifier(random_state=RANDOM_STATE)
clf.fit(X_train_embeddings, y_train_encoded)
    
# Predict and evaluate
y_pred = clf.predict(X_test_embeddings)
accuracy = accuracy_score(y_test_encoded, y_pred)
    
print(f"\nAccuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))
    
# Display confusion matrix
ConfusionMatrixDisplay.from_estimator(
        clf, 
        X_test_embeddings, 
        y_test_encoded,
        display_labels=le.classes_
    )
plt.title('Confusion Matrix')
plt.show()

#%%
from gensim.models import doc2vec

model_doc=doc2vec.Doc2Vec(vector_size=100,min_count=2,epochs=10)
model_doc.build_vocab(sentences)
