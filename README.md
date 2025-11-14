# CMSC 671 Project

A data analysis project for processing and analyzing multiple datasets.

## Project Structure

```
cmsc-671-project/
├── data_analysis/      # Analysis scripts for each dataset
│   ├── imdb.py        # IMDB dataset analysis
│   ├── legal.py       # LEGAL dataset analysis
│   └── mdr.py         # MDR dataset analysis
├── dataset/           # Raw datasets
│   ├── IMDB.csv
│   ├── LEGAL.csv
│   └── MDR.csv
├── processed_dataset/ # Processed data outputs
├── main.py           # Main entry point
└── requirements.txt  # Python dependencies
```

## Setup

1. Clone the repository with Git LFS:

```bash
git lfs install
git clone <repository-url>
cd cmsc-671-project
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Note: This project uses Git LFS (Large File Storage) to manage large dataset files. Make sure you have Git LFS installed before cloning.

## Usage

Run individual dataset analysis scripts:

```bash
# Analyze IMDB dataset
python data_analysis/imdb.py

# Analyze LEGAL dataset
python data_analysis/legal.py

# Analyze MDR dataset
python data_analysis/mdr.py
```

## Datasets

- **IMDB**: Movie review sentiment dataset
- **LEGAL**: Legal text dataset
- **MDR**: Medical document retrieval dataset

## Requirements

- Python 3.x
- pandas
