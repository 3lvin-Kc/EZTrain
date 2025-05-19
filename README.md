# EasyTrain CLI: NLP Model Training on Hugging Face

EasyTrain CLI is a command-line tool that helps you quickly train NLP models using your own datasets and Hugging Face's infrastructure. It supports multiple dataset formats, customizable hyperparameters, and seamless integration with Hugging Face Hub.

---

## Features

- **Train BERT and DistilBERT models** for NLP tasks (e.g., classification)
- **Supports CSV, JSON, and Parquet datasets**
- **Customizable hyperparameters** (epochs, learning rate, batch size)
- **Automatic dataset validation and upload** to Hugging Face
- **Progress monitoring** and result saving
- **API token management** via CLI or config file

---

## Installation

1. Clone this repository.
2. Install dependencies:
    
    pip install -r requirements.txt
  

---

## Getting Started

### 1. Prepare Your Dataset

- Supported formats: CSV, JSON, Parquet
- Must contain at least a text column (default: `text`)
- For classification tasks, a label column is recommended

Example CSV:
```csv
text,label
"This is a positive review",1
"This is a negative review",0

### 2. Set Up Your Hugging Face API Token
- Create a Hugging Face account
- Generate an access token
- You can provide the token via:
  - --api-token CLI option
  - Or save it in a config file at ~/.easytrain/config.json :
    ```json
    {
      "api_token": "your_hf_token_here"
    }
     ```
    ```
## Usage
### Basic Example
### Specify Dataset Format
### Custom Text and Label Columns
### Set Hyperparameters
### Save Results to a Custom File
## Advanced Usage
- Change Model:
- Specify Task:
- Use Parquet Dataset:
## Output
- Results (including metrics and model URL) are saved to the specified output file (default: results.json ).
## Troubleshooting
- Missing API Token: Make sure to provide --api-token or set it in ~/.easytrain/config.json .
- Dataset Validation Failed: Ensure your dataset has the required columns (default: text , and label for classification).
- Other Errors: Check the log output for details.
## License
MIT License

## Contributing
Pull requests and suggestions are welcome!