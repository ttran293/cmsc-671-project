import re
import pandas as pd

df = pd.read_csv('../dataset/LEGAL.csv')

#This function cleans the "case_text" column of the dataset to prep for summarization and sentiment analysis
def clean_case_text(text: str) -> str:
    if pd.isna(text):
        return""

    text = str(text)

    #Remove the citations from case_text
    text = re.sub(r"\[\d{4}\]\s+[A-Z]+\s+\d+", " ", text)
    text = re.sub(r"\(\d{4}\)\s*\d+\s+[A-Z]+\s*\d*", " ", text)

    #Remove any paragrpah markers
    text = re.sub(r"at\s+\[\d+\]", " ", text, flags=re.IGNORECASE)

    #Make all lowercase and remove quotes
    text = text.lower()
    text = text.replace('"', " ")

    #Remove meaningless punctuation and normalize spacing
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


#This function will keep the original raw case text and put it in a new column
#Called "raw_case_text" and save the cleaned case text to a new column "clean_case_text"
#To later be used in the summarization and sentiment anlysis
def clean_legal_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['case_text']).copy()
    df["raw_case_text"] = df["case_text"]
    df["clean_case_text"] = df["case_text"].apply(clean_case_text)
    return df


#This function calls cleaning functions for the dataset to prepare it for summarization and sentiment analysis
#A new cleaned dataset in the form of csv will be saved to the processed_dataset folder
def main():
    cleaned_df = clean_legal_dataset(df)
    cleaned_df.to_csv('../processed_dataset/LEGAL_cleaned.csv', index=False)

    #print the first five clean_case_text to review
    for i, text in cleaned_df["clean_case_text"].head(5).items():
        print(f"\n Case {i} ")
        print(text)


if __name__ == '__main__':
    main()

