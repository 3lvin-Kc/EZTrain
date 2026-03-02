# EasyTrain CLI: NLP Model Training on Hugging Face

EasyTrain CLI is a command-line tool that helps you quickly train NLP models using your own datasets and Hugging Face's infrastructure. It supports multiple dataset formats, customizable hyperparameters, and seamless integration with Hugging Face Hub.

---

## Features

- **Multiple Models**: BERT, DistilBERT, RoBERTa, ALBERT, ELECTRA, DeBERTa, BERTweet
- **Multiple NLP Tasks**: Classification, NER (Named Entity Recognition), Sentiment Analysis, QA
- **Supports CSV, JSON, and Parquet datasets**
- **Customizable hyperparameters** (epochs, learning rate, batch size, warmup, weight decay)
- **FP16 mixed precision training** support
- **Dry-run mode** for dataset validation without training
- **Push to Hugging Face Hub** - automatically upload trained models
- **Automatic dataset validation and upload** to Hugging Face
- **Progress monitoring** and result saving
- **API token management** via CLI or config file

---

## Installation

1. Clone this repository.
2. Install dependencies:
   
   ```
   pip install -r requirements.txt
   ```

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
```

### 2. Set Up Your Hugging Face API Token

- Create a Hugging Face account
- Generate an access token
- You can provide the token via:
  - `--api-token` CLI option
  - Or save it in a config file at `~/.easytrain/config.json`:
    ```json
    {
      "api_token": "your_hf_token_here"
    }
    ```

---

## Usage

### Basic Training

```bash
python cli.py --dataset data.csv --api-token YOUR_TOKEN
```

### Specify Dataset Format

```bash
python cli.py --dataset data.json --file-format json --api-token YOUR_TOKEN
```

### Custom Text and Label Columns

```bash
python cli.py --dataset data.csv --text-column content --label-column category --api-token YOUR_TOKEN
```

### Set Hyperparameters

```bash
python cli.py --dataset data.csv \
  --epochs 5 \
  --lr 3e-5 \
  --batch-size 32 \
  --warmup-ratio 0.2 \
  --weight-decay 0.01 \
  --api-token YOUR_TOKEN
```

### Train Different Models

```bash
python cli.py --dataset data.csv --model roberta --api-token YOUR_TOKEN
python cli.py --dataset data.csv --model albert --api-token YOUR_TOKEN
python cli.py --dataset data.csv --model electra --api-token YOUR_TOKEN
```

### Train Different Tasks

```bash
python cli.py --dataset data.csv --task classification --api-token YOUR_TOKEN
python cli.py --dataset data.csv --task ner --api-token YOUR_TOKEN
python cli.py --dataset data.csv --task sentiment --api-token YOUR_TOKEN
```

### Use FP16 Mixed Precision

```bash
python cli.py --dataset data.csv --fp16 --api-token YOUR_TOKEN
```

### Dry Run (Validate Only)

```bash
python cli.py --dataset data.csv --dry-run --api-token YOUR_TOKEN
```

### Push Model to Hugging Face Hub

```bash
python cli.py --dataset data.csv --push-to-hub --hub-repo-id username/my-model --api-token YOUR_TOKEN
```

### Custom Train/Validation Split

```bash
python cli.py --dataset data.csv --train-split 0.8 --api-token YOUR_TOKEN
```

### Save Results to Custom File

```bash
python cli.py --dataset data.csv --output my_results.json --api-token YOUR_TOKEN
```

---

## CLI Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dataset` | str | required | Path to dataset file or HF dataset name |
| `--file-format` | str | csv | Dataset file format (csv, json, parquet) |
| `--text-column` | str | text | Name of the text column |
| `--label-column` | str | None | Name of the label column |
| `--task` | str | classification | NLP task (classification, ner, sentiment, qa) |
| `--model` | str | bert | Model to train (bert, distilbert, roberta, albert, electra, deberta, bertweet) |
| `--api-token` | str | None | Hugging Face API token |
| `--epochs` | int | 3 | Number of training epochs |
| `--lr` | float | 2e-5 | Learning rate |
| `--batch-size` | int | 16 | Batch size |
| `--warmup-ratio` | float | 0.1 | Warmup ratio |
| `--weight-decay` | float | 0.01 | Weight decay |
| `--logging-steps` | int | 10 | Logging steps |
| `--eval-steps` | int | 50 | Evaluation steps |
| `--train-split` | float | 0.9 | Train split ratio |
| `--seed` | int | 42 | Random seed |
| `--fp16` | flag | False | Use FP16 mixed precision |
| `--dry-run` | flag | False | Validate dataset without training |
| `--push-to-hub` | flag | False | Push model to Hugging Face Hub |
| `--hub-repo-id` | str | None | Hub repository ID for pushing model |
| `--output` | str | results.json | Output file for results |

---

## Output

Results (including metrics and model URL) are saved to the specified output file (default: `results.json`).

Example output:
```json
{
  "status": "completed",
  "model_url": "https://huggingface.co/username/my-model",
  "metrics": {
    "eval_accuracy": 0.92,
    "eval_loss": 0.15,
    "eval_f1": 0.91
  },
  "hyperparameters": {
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16
  },
  "task": "text-classification",
  "model": "bert-base-uncased"
}
```

---

## Testing

```bash
pytest tests/
```

---

## Troubleshooting

- **Missing API Token**: Make sure to provide `--api-token` or set it in `~/.easytrain/config.json`.
- **Dataset Validation Failed**: Ensure your dataset has the required columns (default: `text`, and `label` for classification).
- **Other Errors**: Check the log output for details.

---

## License

MIT License

## Contributing

Pull requests and suggestions are welcome!
