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

sampled_df = df.sample(n=3300, random_state=42).reset_index(drop=True)

train_df = sampled_df.iloc[:3000][['sentence']]
test_df = sampled_df.iloc[3000:][['sentence']]

print(f"Train set: {len(train_df)} samples")
print(f"Test set: {len(test_df)} samples")

dataset_folder = os.path.join(os.path.dirname(__file__), '..', 'clean_dataset', 'MDR')
os.makedirs(dataset_folder, exist_ok=True)

train_df.to_csv(os.path.join(dataset_folder, 'MDR_train.csv'), index=False, encoding='utf-8-sig')
test_df.to_csv(os.path.join(dataset_folder, 'MDR_test.csv'), index=False, encoding='utf-8-sig')
