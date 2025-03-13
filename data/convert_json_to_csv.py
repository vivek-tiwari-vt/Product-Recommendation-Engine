import pandas as pd

# Convert Reviews JSON to CSV
df = pd.read_json('../data/Electronics.json', lines=True)
df = df[['reviewerID', 'asin', 'overall', 'reviewText', 'summary']]
df.columns = ['user_id', 'product_id', 'rating', 'review_text', 'review_summary']
df.to_csv('interactions.csv', index=False)