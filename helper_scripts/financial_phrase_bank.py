import pandas as pd
from datasets import load_dataset
import os

ds = load_dataset("lmassaron/FinancialPhraseBank")

train_df = pd.DataFrame(ds['train'])
validation_df = pd.DataFrame(ds['validation'])
test_df = pd.DataFrame(ds['test'])

def clean_dataframe(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace('``', "''", regex=False)
    return df

combined_df = pd.concat([train_df, validation_df, test_df], ignore_index=True)
combined_df = clean_dataframe(combined_df)

sampled_df = combined_df.sample(n=3300, random_state=42).reset_index(drop=True)

sampled_df = sampled_df[['sentence']]

train_df = sampled_df.iloc[:3000]
test_df = sampled_df.iloc[3000:]

print(f"Train set: {len(train_df)} samples")
print(f"Test set: {len(test_df)} samples")

dataset_folder = os.path.join(os.path.dirname(__file__), '..', 'clean_dataset', 'FinancialPhraseBank')
os.makedirs(dataset_folder, exist_ok=True)

train_df.to_csv(os.path.join(dataset_folder, 'FinancialPhraseBank_train.csv'), index=False, encoding='utf-8-sig')
test_df.to_csv(os.path.join(dataset_folder, 'FinancialPhraseBank_test.csv'), index=False, encoding='utf-8-sig')