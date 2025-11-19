import json
import re
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
from langchain_community.llms import DeepInfra
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

analyzer = SentimentIntensityAnalyzer()

fpb_df = pd.read_csv('../clean_dataset/FinancialPhraseBank_train.csv', encoding='utf-8')
imdb_df = pd.read_csv('../clean_dataset/IMDB_train.csv', encoding='utf-8')
mdr_df = pd.read_csv('../clean_dataset/MDR_train.csv', encoding='utf-8')

load_dotenv()
os.environ["DEEPINFRA_API_TOKEN"] = os.getenv('DEEPINFRA_API_TOKEN')

llm = DeepInfra(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct")
model = "siebert/sentiment-roberta-large-english"
classifier = pipeline("sentiment-analysis", model=model, truncation=True, max_length=512)

def estimate_tokens(text):
    words = len(text.split())
    return int(words * 1.3)

def get_preserve_prompt(sentence):
    prompt = f"""You must respond with ONLY valid JSON. No code, no comments, no extra text.

    Text to summarize: {sentence}

    Requirements:
    - Keep all important information and details in the summary.
    - Summarize the following sentence while preserving its original tone, wording style, and sentiment.

    Respond with this exact JSON structure:
    {{"result": "your summary here"}}"""
    return prompt

def get_no_preserve_prompt(sentence):
    prompt = f"""You must respond with ONLY valid JSON. No code, no comments, no extra text.

    Text to summarize: {sentence}

    Requirements:
    - Keep all important information and details in the summary.
    - Summarize the following sentence to a neutral tone.

    Respond with this exact JSON structure:
    {{"result": "your summary here"}}"""
    return prompt

def get_summary_prompt(sentence):
    prompt = f"""You must respond with ONLY valid JSON. No code, no comments, no extra text.

    Text to summarize: {sentence}

    Requirements:
    - Keep all important information and details in the summary.

    Respond with this exact JSON structure:
    {{"result": "your summary here"}}"""
    return prompt

def get_vader_sentiment(sentence):
    scores = analyzer.polarity_scores(sentence)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return label, compound

def get_roberta_sentiment(sentence):
    result = classifier(sentence)
    return result[0]['label'], result[0]['score']

def decide_mode(sentence):
    _, score = get_roberta_sentiment(sentence)
    if score >= 0.7:
        return "preserve"
    return "neutral"


def validate_summary(mode, orig_label, orig_score, sum_label, sum_score, sum_vader_label, sum_vader_score):
    if mode == "neutral":
        if sum_label in ("NEUTRAL") or sum_vader_label in ("neutral"):
            return True
        return False

    if mode == "preserve":
        if orig_label in ("POSITIVE", "NEGATIVE"):
            if sum_label != orig_label:
                return False
            if sum_score < 0.5:
                return False
        return True

    return True

def extract_result_from_raw(raw, fallback):
    raw_clean = str(raw).strip()

    raw_no_fence = re.sub(r"```.*?```", "", raw_clean, flags=re.DOTALL)

    try:
        data = json.loads(raw_no_fence)
        if isinstance(data, dict) and "result" in data:
            return str(data["result"]).strip()
    except Exception:
        pass

    candidate_objs = re.findall(r'\{[^{}]*"result"[^{}]*\}', raw_no_fence, flags=re.DOTALL)

    last_good = None
    for obj in candidate_objs:
        try:
            data = json.loads(obj)
            if isinstance(data, dict) and "result" in data:
                last_good = str(data["result"]).strip()
        except Exception:
            continue

    if last_good is not None:
        return last_good

    m = re.findall(r'"result"\s*:\s*"([^"]+)"', raw_no_fence)
    if m:
        return m[-1].strip()

    return fallback


def generate_summary(df, mode_label):
    results = []
    for idx, row in df.iterrows():
        sentence = row['sentence']

        if mode_label == 'auto':
            effective_mode = decide_mode(sentence)
        else:
            effective_mode = mode_label

        if effective_mode == 'preserve':
            prompt = get_preserve_prompt(sentence)
        elif effective_mode == 'neutral':
            prompt = get_no_preserve_prompt(sentence)
        else:
            prompt = get_summary_prompt(sentence)

        max_tokens = estimate_tokens(sentence) * 3 + 200

        llm.model_kwargs = {
            "temperature": 0.2,
            "repetition_penalty": 1.2,
            "max_new_tokens": max_tokens,
            "top_p": 0.9,
        }
        # print("\n\n\n--------------------------------")
        # print("Prompt:")
        # print(prompt)
        # print("\n\n\n--------------------------------")
        raw = llm.invoke(prompt)
        # print("\n\n\n--------------------------------")
        # print(raw)
        # print("\n\n\n--------------------------------")
        raw_clean = str(raw).strip()
        summarized_text = extract_result_from_raw(raw_clean, sentence)

    

        original_sentiment, original_score = get_roberta_sentiment(sentence)
        summarized_roberta_sentiment, summarized_roberta_score = get_roberta_sentiment(summarized_text)
        original_vader_sentiment, original_vader_score = get_vader_sentiment(sentence)
        summarized_vader_sentiment, summarized_vader_score = get_vader_sentiment(summarized_text)
        is_valid = validate_summary(effective_mode, original_sentiment, original_score, summarized_roberta_sentiment, summarized_roberta_score, summarized_vader_sentiment, summarized_vader_score)
        results.append({
            'original_text': sentence,
            'vader_original_sentiment': original_vader_sentiment,
            'vader_original_score': original_vader_score,
            'roberta_original_sentiment': original_sentiment,
            'roberta_original_score': original_score,
            'summarized_text': summarized_text,
            'vader_summarized_sentiment': summarized_vader_sentiment,
            'vader_summarized_score': summarized_vader_score,
            'roberta_summarized_sentiment': summarized_roberta_sentiment,
            'roberta_summarized_score': summarized_roberta_score,
            'mode_used': effective_mode,
            'is_valid': is_valid,
        })

    return results

if __name__ == "__main__":
    fpb_df = fpb_df.head(5)
    imdb_df = imdb_df.head(5)
    mdr_df = mdr_df.head(5)

    all_results = []

    fpb_results = generate_summary(fpb_df, 'neutral')
    all_results.extend(fpb_results)

    imdb_results = generate_summary(imdb_df, 'auto')
    all_results.extend(imdb_results)

    mdr_results = generate_summary(mdr_df, 'preserve')
    all_results.extend(mdr_results)

    result_df = pd.DataFrame(all_results)

    output_folder = os.path.join(os.path.dirname(__file__), '.', 'summarized_dataset')
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, 'all_summaries.csv')
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
