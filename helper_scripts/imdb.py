import pandas as pd
import re
import os

df = pd.read_csv('../original_dataset/IMDB.csv', usecols=['review', 'sentiment'])


def clean_dataframe(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace('``', "", regex=False)
            df[col] = df[col].str.replace(r'<[^>]+>', ' ', regex=True)
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            df[col] = df[col].str.strip()
    return df


df = clean_dataframe(df)

sampled_df = df.sample(n=3300, random_state=42).reset_index(drop=True)

sampled_df = sampled_df.rename(columns={'review': 'sentence'})

train_df = sampled_df.iloc[:3000][['sentence']]
test_df = sampled_df.iloc[3000:][['sentence']]

print(f"Train set: {len(train_df)} samples")
print(f"Test set: {len(test_df)} samples")

dataset_folder = os.path.join(os.path.dirname(__file__), '..', 'clean_dataset', 'IMDB')
os.makedirs(dataset_folder, exist_ok=True)

train_df.to_csv(os.path.join(dataset_folder, 'IMDB_train.csv'), index=False)
test_df.to_csv(os.path.join(dataset_folder, 'IMDB_test.csv'), index=False)