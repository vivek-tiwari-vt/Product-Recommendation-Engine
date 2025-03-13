from gensim.models import Word2Vec
import pandas as pd
import numpy as np

class NLPFiltering:
    def __init__(self):
        self.word2vec = None
    
    def train(self, metadata_df):
        sentences = metadata_df['processed_text'].str.split().tolist()
        self.word2vec = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)
    
    def generate_embeddings(self, metadata_df):
        def get_embedding(text):
            words = text.split()
            vectors = [self.word2vec.wv[word] for word in words if word in self.word2vec.wv]
            return np.mean(vectors, axis=0) if len(vectors) > 0 else np.zeros(self.word2vec.vector_size)
        
        metadata_df['embeddings'] = metadata_df['processed_text'].apply(get_embedding)
        return metadata_df
    
    def save_model(self, path):
        self.word2vec.save(path)