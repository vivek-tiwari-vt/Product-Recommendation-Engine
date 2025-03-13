from flask import Flask, request, jsonify
from src.hybrid_engine import HybridRecommender
import pandas as pd
import joblib
from gensim.models import Word2Vec

app = Flask(__name__)

# Load Models
cf_model = joblib.load('../models/cf_model.pkl')
nlp_model = Word2Vec.load('../models/word2vec.model')
metadata = pd.read_csv('../data/product_metadata.csv')

# Initialize Recommender
recommender = HybridRecommender(cf_model, nlp_model, metadata)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    try:
        recommendations = recommender.recommend(user_id)
        return jsonify({"recommendations": recommendations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)