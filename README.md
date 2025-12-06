# CMSC-671 Project  
**Affective Summarization: Deciding When to Preserve Sentiment**

Deciding when to preserve sentiment during summarization is a key challenge for large language models (LLMs). In some domains containing affective or emotionally charged data, maintaining sentiment is essential; in others, a neutral summary is preferred.  
This project develops a system that automatically determines when sentiment should be preserved and generates summaries accordingly.

We explore transformer-based techniques for sentiment detection that integrate extractive and generative approaches. Multiple datasets across different domains are evaluated, and we test the system using language models of varying capability. The findings highlight trade-offs in affective summarization and inform best practices for deploying summarization systems across diverse tasks.

---

## Prerequisites

- Python 3.11+  
- Git with Git LFS  
- Hugging Face account with access to **LLaMA 3.1-8B-Instruct**  
- DeepInfra API token  
- At least **32 GB RAM**  
- CUDA-capable GPU (recommended for training)  
- Can be run on **Google Colab** (L4 or A100 GPU recommended)

---

## Getting API Tokens

### DeepInfra
- Sign up at: https://deepinfra.com  
- Retrieve your API token  
- Note: Not required when testing the finetuned model

### Hugging Face
1. Create an account: https://huggingface.co  
2. Request access to **LLaMA 3.1-8B-Instruct**  
3. Generate an access token:  
   **Settings → Access Tokens**

---

## Usage

### Self-Taught “Naive” Approach

A Hugging Face key and model access permission are required.

Project notebooks (in execution order):

1. **generate_gold_output_script.ipynb**  
   Creates the “gold” output used for finetuning.

2. **create_finetune_test.ipynb**  
   Converts CSV gold output to JSONL format for finetuning.

3. **finetune.ipynb**  
   Finetunes the model and uploads it to Hugging Face.

4. **evaluation.ipynb**  
   Runs evaluation and records metric scores on the test set.

5. **evaluation_analysis.ipynb**  
   Produces analyses and visualizations.

All outputs and images are stored in the same folder.

---

### Neurosymbolic Approach

Requires Hugging Face access to the model.

Notebook workflow:

1. **neurosymbolic.ipynb**  
   Integrates spaCy + ConceptNet for a classification + summarization pipeline.

2. **evaluation_analysis.ipynb**  
   Produces analyses and visualizations.

All outputs and images are stored in the same folder.

---

## Datasets

This project uses three datasets:

### Financial Phrase Bank (FPB)
- Financial news sentences with sentiment labels  
- Train: ~1,800 | Test: ~486  
- Sentiment: Positive, Neutral, Negative

### IMDB Movie Reviews
- Long-form movie reviews  
- Train: ~20,000 | Test: ~5,000  
- Sentiment: Positive, Negative

### Mental Health Reddit (MDR)
- Mental health support posts  
- Train: ~232,000 | Test: ~58,000  
- Contains diverse emotional content  
- Preprocessed version stored separately on GitHub

---

## Acknowledgments

- Meta AI for LLaMA 3.1  
- Hugging Face for the transformers library  
- UMBC HPCC for compute resources  
- Google Colab Pro – Student Version  
