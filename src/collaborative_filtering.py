import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_split
import joblib

class CollaborativeFiltering:
    def __init__(self):
        self.model = SVD(n_factors=50, reg_all=0.02, verbose=True)
    
    def train(self, interactions_df):
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(interactions_df[['user_id', 'product_id', 'rating']], reader)
        trainset, _ = surprise_split(data, test_size=0.2)
        self.model.fit(trainset)
    
    def save_model(self, path):
        joblib.dump(self.model, path)