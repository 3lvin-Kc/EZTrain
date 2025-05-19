import click
import json
import logging
from huggingface_hub import login
from pathlib import Path

from dataset_utils import validate_dataset, upload_dataset
from train_utils import start_training, monitor_training
from config_utils import load_api_token

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_MAP = {
    'bert': 'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased'
}
DEFAULT_HYPERPARAMS = {
    'learning_rate': 2e-5,
    'num_train_epochs': 3,
    'per_device_train_batch_size': 16
}

@click.command()
@click.option('--dataset', required=True, type=str, help='Path to dataset file')
@click.option('--file-format', default='csv', type=click.Choice(['csv', 'json', 'parquet']), help='Dataset file format')
@click.option('--text-column', default='text', type=str, help='Name of the text column')
@click.option('--label-column', default=None, type=str, help='Name of the label column (optional)')
@click.option('--task', default='classification', type=str, help='NLP task (e.g., classification)')
@click.option('--model', default='bert', type=click.Choice(['bert', 'distilbert']), help='Model to train')
@click.option('--api-token', type=str, help='Hugging Face API token')
@click.option('--epochs', type=int, help='Number of training epochs')
@click.option('--lr', type=float, help='Learning rate')
@click.option('--batch-size', type=int, help='Batch size')
@click.option('--output', default='results.json', type=str, help='Output file for results')
def train(dataset, file_format, text_column, label_column, task, model, api_token, epochs, lr, batch_size, output):
    """CLI tool to train NLP models on Hugging Face."""
    try:
        # Load API token from flag or config
        api_token = load_api_token(api_token)
        if not api_token:
            raise click.UsageError("API token required. Use --api-token or set in ~/.easytrain/config.json")

        # Login to Hugging Face
        login(token=api_token)

        # Validate dataset
        if not validate_dataset(dataset, file_format, text_column, label_column):
            raise click.UsageError(f"Dataset validation failed. Ensure it has a '{text_column}' column.")

        # Upload dataset
        import time
        repo_name = f"dataset-{int(time.time())}"
        dataset_url = upload_dataset(dataset, repo_name, api_token, file_format)

        # Prepare hyperparameters
        hyperparams = DEFAULT_HYPERPARAMS.copy()
        if epochs:
            hyperparams['num_train_epochs'] = epochs
        if lr:
            hyperparams['learning_rate'] = lr
        if batch_size:
            hyperparams['per_device_train_batch_size'] = batch_size

        # Start training
        model_name = MODEL_MAP[model]
        job_id = start_training(dataset_url, model_name, task, hyperparams, api_token)

        # Monitor training
        result = monitor_training(job_id, api_token)

        # Save results
        with open(output, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to {output}")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise click.Abort()

if __name__ == '__main__':
    train()