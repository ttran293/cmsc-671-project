# CMSC-671 Project

Deciding when to preserve sentiment during summarization is a challenge with Large Language Models (LLMs). In some domains that have affective data, maintaining sentiment is essential, while in others, a neutral summary is preferred. This study aims to develop a system capable of automatically identifying when sentiment should be preserved and generating summaries accordingly. We explore various transformer-based techniques for sentiment detection that integrate both extractive and generative approaches. We examine multiple datasets across different domains to assess our goal. In addition, we also test our system using different language models with different capability settings. The findings of this study will help to understand the trade-offs in affective summarization and inform best practices for applying summarization systems across diverse tasks.

## Prerequisites

- Python 3.11+
- Git with Git LFS installed
- Hugging Face account with access to LLaMA 3.1-8B-Instruct
- DeepInfra API token (for summary generation not needed for testing)
- At least 32GB RAM
- CUDA-capable GPU (recommended for training)

## Installation

### Linux

1. **Clone the repository:**

```bash
git clone <your-repo-url>
cd cmsc-671-project
```

2. **Create a virtual environment:**

```bash
python3.11 -m venv venv311
```

3. **Activate the virtual environment:**

```bash
source venv311/bin/activate
```

4. **Install dependencies:**

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the project root directory with the following variables:

```env
DEEPINFRA_API_TOKEN=your_deepinfra_token_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

**Getting API Tokens:**

- **DeepInfra:** Sign up at [deepinfra.com](https://deepinfra.com) and get your API token (Will not be needed to test the finetuned model)
- **Hugging Face:**
  1. Sign up at [huggingface.co](https://huggingface.co)
  2. Request access to [LLaMA 3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
  3. Generate a token at [Settings &gt; Access Tokens](https://huggingface.co/settings/tokens)

## Git LFS Setup

This project uses Git LFS to manage large dataset files. Make sure Git LFS is installed and pull the actual files:

```bash
# Install Git LFS (if not already installed)
# Windows: Download from https://git-lfs.github.com/
# Linux: sudo apt-get install git-lfs

# Initialize Git LFS
git lfs install

# Pull the actual dataset files
git lfs pull
```

**Verify files are downloaded:**

```bash
# Check if CSV files contain actual data (not LFS pointers)
head -n 3 test_set/FinancialPhraseBank_test.csv
```

If you see `version https://git-lfs.github.com/spec/v1`, the files are still pointers - run `git lfs pull` again.

## Usage

### How to test: Self-Taught Approach

See environment setup. A Huggingface key is required to test this.
You will also need to request permission to use LLaMA 3.1-8B-Instruct model.

Once the key is obtained, simply add it to the .env file. Run the scripts test_finetuned_result.py to test

Compare base model vs fine-tuned model performance:

```bash
cd prompt_lora_scripts
python test_finetuned_result.py
```

**What it does:**

1. Loads the first sample from each test dataset (FPB, IMDB, MDR)
2. Generates summaries using:
   - Base model: `meta-llama/Llama-3.1-8B-Instruct`
   - Fine-tuned model: `ttran19/llama3-lora-671`
3. Saves comparison results to CSV

**Example output:**

```
Processing FPB - 1/1
Text: Loss after taxes amounted to EUR 1.2 mn compared to a loss of 2.6 mn.
Summary: [Generated summary]

Processing IMDB - 1/1
Text: [Movie review text...]
Summary: [Generated summary]

Processing MDR - 1/1
Text: [Mental health post...]
Summary: [Generated summary]
```
Note: If you don't want to run it, the output is saved in the prompt_lora_scripts\model_comparison_results\model_comparision.csv

## UMBC HPCF Usage

If you're using UMBC's High Performance Computing Cluster:

### Request GPU Node

```bash
# Connect to HPCF
ssh your_username@chip.rs.umbc.edu

# Request GPU node
srun --cluster=chip-gpu \
     --account=user \
     --time=60 \
     --mem=32000 \
     --gres=gpu:1 \
     --pty $SHELL

#activate shell cpu
srun --cluster=chip-cpu \
     --account=user \
     --partition=general \
     --qos=short \
     --time=00:30:00 \
     --mem=4000 \
     --pty $SHELL

# Load Python 3.11
module load Python/3.11.5-GCCcore-13.2.0

# Activate virtual environment
source venv311/bin/activate
```

### Fix Cache Issues

If you encounter "No space left on device" errors:

```bash
# Set custom cache directories
export HF_HOME=/umbc/rs/pi_cmat/users/your_username/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/hub
```

## Datasets

This project uses three datasets:

1. **Financial Phrase Bank (FPB)**

   - Financial news sentences with sentiment labels
   - Train: ~1,800 samples | Test: ~486 samples
   - Sentiment: Positive, Neutral, Negative
2. **IMDB Movie Reviews**

   - Long-form movie reviews
   - Train: ~20,000 samples | Test: ~5,000 samples
   - Sentiment: Positive, Negative
3. **Mental Health Reddit (MDR)**

   - Mental health support posts from Reddit
   - Train: ~232,000 samples | Test: ~58,000 samples
   - Diverse emotional content

## Acknowledgments

- Meta AI for LLaMA 3.1
- Hugging Face for transformers library
- UMBC HPCC for computing resources
