from transformers import pipeline
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv('HUGGINGFACE_TOKEN')

test_fpb_df = pd.read_csv('../test_set/FinancialPhraseBank_test.csv', encoding='utf-8')
test_imdb_df = pd.read_csv('../test_set/IMDB_test.csv', encoding='utf-8')
test_mdr_df = pd.read_csv('../test_set/MDR_test.csv', encoding='utf-8')

finetuned_model_id = "ttran19/llama3-lora-671"
base_model_id = "meta-llama/Llama-3.1-8B-Instruct"

sample_fpb = test_fpb_df.head(1)
sample_imdb = test_imdb_df.head(1)
sample_mdr = test_mdr_df.head(1)

print("FPB columns:", test_fpb_df.columns.tolist())
print("IMDB columns:", test_imdb_df.columns.tolist())
print("MDR columns:", test_mdr_df.columns.tolist())

def estimate_tokens(text):
    words = len(text.split())
    return int(words * 1.3)

def generate_summary(text, model_id):
    max_tokens = estimate_tokens(text) * 2 + 200
    prompt = f"Summarize the following text.\n\n{text}"
    
    summarizer = pipeline(
        task="text-generation",   
        model=model_id,
        tokenizer=model_id,
        max_new_tokens=max_tokens,
        do_sample=False,
        token=hf_token
    )

    outputs = summarizer(prompt)
    summary = outputs[0]["generated_text"]
    return summary

def process_samples(df, dataset_name, model_id):
    results = []
    for idx, row in df.iterrows():
        print(f"Processing {dataset_name} - {idx+1}/{len(df)}")
        text = row['sentence']
        print(f"Text: {text}")
        summary = generate_summary(text, model_id)
        print(f"Summary: {summary}")
        results.append({
            'dataset': dataset_name,
            'original_text': text,
            'summary': summary
        })
    return results

if __name__ == "__main__":
    all_results = []
    
    all_results.extend(process_samples(sample_fpb, 'FPB', base_model_id))
    all_results.extend(process_samples(sample_imdb, 'IMDB', base_model_id))
    all_results.extend(process_samples(sample_mdr, 'MDR', base_model_id))
    
    base_results_df = pd.DataFrame(all_results)
    base_results_df['model'] = 'base'
    
    finetuned_results = []
    
    finetuned_results.extend(process_samples(sample_fpb, 'FPB', finetuned_model_id))
    finetuned_results.extend(process_samples(sample_imdb, 'IMDB', finetuned_model_id))
    finetuned_results.extend(process_samples(sample_mdr, 'MDR', finetuned_model_id))
    
    finetuned_results_df = pd.DataFrame(finetuned_results)
    finetuned_results_df['model'] = 'finetuned'
    
    final_results_df = pd.concat([base_results_df, finetuned_results_df], ignore_index=True)
    
    output_folder = os.path.join(os.path.dirname(__file__), '.', 'model_comparison_results')
    os.makedirs(output_folder, exist_ok=True)
    
    output_file = os.path.join(output_folder, 'model_comparison.csv')
    final_results_df.to_csv(output_file, index=False, encoding='utf-8-sig')