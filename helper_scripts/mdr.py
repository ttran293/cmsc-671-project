import pandas as pd
import re
import os

df = pd.read_csv('../original_dataset/MDR.csv', usecols=['title', 'selftext'])

df = df[df['selftext'] != '[removed]']
df = df.dropna(subset=['selftext'])

def clean_dataframe(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            df[col] = df[col].str.strip()
    return df

df = clean_dataframe(df)

df = df.rename(columns={'selftext': 'sentence'})

print(df.head())
train_size = int(len(df) * 0.9)
train_df = df.iloc[:train_size][['sentence']]
test_df = df.iloc[train_size:][['sentence']]

dataset_folder = os.path.join(os.path.dirname(__file__), '..', 'clean_dataset')
os.makedirs(dataset_folder, exist_ok=True)

train_df.to_csv(os.path.join(dataset_folder, 'MDR_train.csv'), index=False, encoding='utf-8-sig')
test_df.to_csv(os.path.join(dataset_folder, 'MDR_test.csv'), index=False, encoding='utf-8-sig')

# print(f"Train set: {len(train_df)} samples (90%)")
# print(f"Test set: {len(test_df)} samples (10%)")
# print(f"Saved to clean_dataset/")
