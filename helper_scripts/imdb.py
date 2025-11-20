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

train_size = int(len(df) * 0.9)
train_df = df.iloc[:train_size]
train_df = train_df.rename(columns={'review': 'sentence'})['sentence']
test_df = df.iloc[train_size:]
test_df = test_df.rename(columns={'review': 'sentence'})['sentence']

dataset_folder = os.path.join(os.path.dirname(__file__), '..', 'clean_dataset')
os.makedirs(dataset_folder, exist_ok=True)

train_df.to_csv(os.path.join(dataset_folder, 'IMDB_train.csv'), index=False)
test_df.to_csv(os.path.join(dataset_folder, 'IMDB_test.csv'), index=False)