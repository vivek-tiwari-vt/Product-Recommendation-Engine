import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class HybridRecommender:
    def __init__(self, cf_model, nlp_model, metadata):
        self.cf_model = cf_model
        self.nlp_model = nlp_model
        self.metadata = metadata
        self.cosine_sim = cosine_similarity(np.vstack(metadata['embeddings'].values))
    
    def recommend(self, user_id, top_n=10):
        # Collaborative Filtering Scores
        user_ratings = [self.cf_model.predict(uid=user_id, iid=pid) for pid in self.metadata['product_id']]
        cf_scores = sorted([(pred.iid, pred.est) for pred in user_ratings], key=lambda x: x[1], reverse=True)[:top_n]
        
        # Content-Based Scores
        user_products = self.metadata[self.metadata['user_id'] == user_id]['product_id'].unique()
        content_scores = []
        for product_id in user_products:
            idx = self.metadata[self.metadata['product_id'] == product_id].index[0]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
            content_scores.extend([(self.metadata.iloc[i]['product_id'], score) for i, score in sorted_scores])
        
        # Combine Scores
        hybrid_scores = {}
        for pid, score in cf_scores:
            hybrid_scores[pid] = score * 0.6
        for pid, score in content_scores:
            hybrid_scores[pid] = hybrid_scores.get(pid, 0) + score * 0.4
        
        return sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]