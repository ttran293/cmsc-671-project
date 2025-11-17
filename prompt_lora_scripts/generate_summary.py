import json
import re
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
from langchain_community.llms import DeepInfra

fpb_df = pd.read_csv('../clean_dataset/FinancialPhraseBank_train.csv', encoding='utf-8')
imdb_df = pd.read_csv('../clean_dataset/IMDB_train.csv', encoding='utf-8')
mdr_df = pd.read_csv('../clean_dataset/MDR_train.csv', encoding='utf-8')

load_dotenv()
os.environ["DEEPINFRA_API_TOKEN"] = os.getenv('DEEPINFRA_API_TOKEN')

llm = DeepInfra(model_id="meta-llama/Meta-Llama-3-8B-Instruct")

def estimate_tokens(text):
    words = len(text.split())
    return int(words * 1.3)

def get_preserve_prompt(sentence):
    prompt = f"""You must respond with ONLY valid JSON. No code, no comments, no extra text.

    Text to summarize: {sentence}

    Requirements:
    - Keep all important information and details in the summary.
    - Preserve the sentiment of the original text.

    Respond with this exact JSON structure:
    {{"result": "your summary here", "explanation": "how you derived it", "mode": "preserve"}}"""
    return prompt

def get_no_preserve_prompt(sentence):
    prompt = f"""You must respond with ONLY valid JSON. No code, no comments, no extra text.

    Text to summarize: {sentence}

    Requirements:
    - Keep all important information and details in the summary.
    - Do not preserve the sentiment of the original text.

    Respond with this exact JSON structure:
    {{"result": "your summary here", "explanation": "how you derived it", "mode": "neutral"}}"""
    return prompt

def get_summary_prompt(sentence):
    prompt = f"""You must respond with ONLY valid JSON. No code, no comments, no extra text.

    Text to summarize: {sentence}

    Requirements:
    - Keep all important information and details in the summary.

    Respond with this exact JSON structure:
    {{"result": "your summary here", "explanation": "how you derived it", "mode": "unspecified"}}"""
    return prompt

def generate_summary(df, mode_label, domain_name):
    results = []
    for idx, row in df.iterrows():
        sentence = row['sentence']

        if mode_label == 'preserve':
            prompt = get_preserve_prompt(sentence)
        elif mode_label == 'neutral':
            prompt = get_no_preserve_prompt(sentence)
        else:
            prompt = get_summary_prompt(sentence)

        max_tokens = estimate_tokens(sentence) * 2 + 100

        llm.model_kwargs = {
            "temperature": 0.2,
            "repetition_penalty": 1.2,
            "max_new_tokens": max_tokens,
            "top_p": 0.9,
        }

        raw = llm.invoke(prompt)
        raw_clean = str(raw).strip()

        try:
            data = json.loads(raw_clean)
        except json.JSONDecodeError:
            data = None
            match = re.search(r'\{.*\}', raw_clean, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    data = None
            if data is None:
                result_match = re.search(r'["\']?result["\']?\s*[:=]\s*["\']([^"\']+)["\']', raw_clean, re.IGNORECASE)
                summarized_text = result_match.group(1).strip() if result_match else sentence
                explanation = ""
                mode_from_model = mode_label
            else:
                summarized_text = data.get("result", sentence).strip()
                explanation = data.get("explanation", "").strip()
                mode_from_model = data.get("mode", mode_label).strip()
        else:
            summarized_text = data.get("result", sentence).strip()
            explanation = data.get("explanation", "").strip()
            mode_from_model = data.get("mode", mode_label).strip()

        results.append({
            'original_text': sentence,
            'summarized_text': summarized_text,
            'mode': mode_from_model,
            'intended_mode': mode_label,
            'domain': domain_name,
            'explanation': explanation,
        })

    return results

if __name__ == "__main__":
    fpb_df = fpb_df.head(1)
    imdb_df = imdb_df.head(1)
    mdr_df = mdr_df.head(1)

    all_results = []

    fpb_results = generate_summary(fpb_df, 'neutral', 'financial_phrasebank')
    all_results.extend(fpb_results)

    imdb_results = generate_summary(imdb_df, 'unspecified', 'imdb')
    all_results.extend(imdb_results)

    mdr_results = generate_summary(mdr_df, 'preserve', 'mdr')
    all_results.extend(mdr_results)

    result_df = pd.DataFrame(all_results)

    output_folder = os.path.join(os.path.dirname(__file__), '.', 'summarized_dataset')
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, 'all_summaries.csv')
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
