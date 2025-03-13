import pandas as pd

# Convert Metadata JSON to CSV
meta_df = pd.read_json('../data/meta_Electronics.json', lines=True)
meta_df = meta_df[['asin', 'title', 'description', 'category']]
meta_df.columns = ['product_id', 'title', 'description', 'category']
meta_df.to_csv('product_metadata.csv', index=False)