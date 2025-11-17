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

train_df = clean_dataframe(train_df)
train_df = train_df['sentence']
# validation_df = clean_dataframe(validation_df)
# test_df = clean_dataframe(test_df)

dataset_folder = os.path.join(os.path.dirname(__file__), '..', 'clean_dataset')
os.makedirs(dataset_folder, exist_ok=True)

train_df.to_csv(os.path.join(dataset_folder, 'FinancialPhraseBank_train.csv'), index=False, encoding='utf-8-sig')
# validation_df.to_csv(os.path.join(dataset_folder, 'FinancialPhraseBank_validation.csv'), index=False, encoding='utf-8-sig')
# test_df.to_csv(os.path.join(dataset_folder, 'FinancialPhraseBank_test.csv'), index=False, encoding='utf-8-sig')