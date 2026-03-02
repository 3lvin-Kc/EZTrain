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
    'distilbert': 'distilbert-base-uncased',
    'roberta': 'roberta-base',
    'albert': 'albert-base-v2',
    'electra': 'google/electra-base-discriminator',
    'deberta': 'microsoft/deberta-base',
    'bertweet': 'vinai/bertweet-base',
}
TASK_MAP = {
    'classification': 'text-classification',
    'ner': 'token-classification',
    'sentiment': 'sentiment-analysis',
    'qa': 'question-answering',
}
DEFAULT_HYPERPARAMS = {
    'learning_rate': 2e-5,
    'num_train_epochs': 3,
    'per_device_train_batch_size': 16,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'logging_steps': 10,
    'eval_steps': 50,
    'save_steps': 50,
    'save_total_limit': 2,
    'load_best_model_at_end': True,
    'evaluation_strategy': 'steps',
}

@click.command()
@click.option('--dataset', required=True, type=str, help='Path to dataset file or HF dataset name')
@click.option('--file-format', default='csv', type=click.Choice(['csv', 'json', 'parquet']), help='Dataset file format')
@click.option('--text-column', default='text', type=str, help='Name of the text column')
@click.option('--label-column', default=None, type=str, help='Name of the label column (optional)')
@click.option('--task', default='classification', type=click.Choice(['classification', 'ner', 'sentiment', 'qa']), help='NLP task')
@click.option('--model', default='bert', type=click.Choice(['bert', 'distilbert', 'roberta', 'albert', 'electra', 'deberta', 'bertweet']), help='Model to train')
@click.option('--api-token', type=str, help='Hugging Face API token')
@click.option('--epochs', type=int, help='Number of training epochs')
@click.option('--lr', type=float, help='Learning rate')
@click.option('--batch-size', type=int, help='Batch size')
@click.option('--output', default='results.json', type=str, help='Output file for results')
@click.option('--warmup-ratio', type=float, default=0.1, help='Warmup ratio')
@click.option('--weight-decay', type=float, default=0.01, help='Weight decay')
@click.option('--logging-steps', type=int, default=10, help='Logging steps')
@click.option('--eval-steps', type=int, default=50, help='Evaluation steps')
@click.option('--train-split', type=float, default=0.9, help='Train split ratio')
@click.option('--seed', type=int, default=42, help='Random seed')
@click.option('--fp16', is_flag=True, help='Use mixed precision training (FP16)')
@click.option('--dry-run', is_flag=True, help='Validate dataset without training')
@click.option('--push-to-hub', is_flag=True, help='Push model to Hugging Face Hub')
@click.option('--hub-repo-id', type=str, help='Hub repository ID for pushing model')
def train(dataset, file_format, text_column, label_column, task, model, api_token, epochs, lr, batch_size, output, warmup_ratio, weight_decay, logging_steps, eval_steps, train_split, seed, fp16, dry_run, push_to_hub, hub_repo_id):
    """CLI tool to train NLP models on Hugging Face."""
    try:
        api_token = load_api_token(api_token)
        if not api_token:
            raise click.UsageError("API token required. Use --api-token or set in ~/.easytrain/config.json")

        login(token=api_token)

        is_hf_dataset = dataset.startswith('hf://') or '/' in dataset
        if is_hf_dataset:
            logger.info(f"Using existing Hugging Face dataset: {dataset}")
            dataset_url = dataset
        else:
            if not validate_dataset(dataset, file_format, text_column, label_column):
                raise click.UsageError(f"Dataset validation failed. Ensure it has a '{text_column}' column.")
            
            if dry_run:
                logger.info("Dry run mode: Dataset validation passed, skipping training.")
                with open(output, 'w') as f:
                    json.dump({'status': 'dry_run', 'dataset_validated': True}, f, indent=2)
                return

            import time
            repo_name = f"dataset-{int(time.time())}"
            dataset_url = upload_dataset(dataset, repo_name, api_token, file_format)

        hyperparams = DEFAULT_HYPERPARAMS.copy()
        hyperparams['num_train_epochs'] = epochs if epochs else hyperparams['num_train_epochs']
        hyperparams['learning_rate'] = lr if lr else hyperparams['learning_rate']
        hyperparams['per_device_train_batch_size'] = batch_size if batch_size else hyperparams['per_device_train_batch_size']
        hyperparams['warmup_ratio'] = warmup_ratio
        hyperparams['weight_decay'] = weight_decay
        hyperparams['logging_steps'] = logging_steps
        hyperparams['eval_steps'] = eval_steps
        hyperparams['save_steps'] = eval_steps
        hyperparams['seed'] = seed

        model_name = MODEL_MAP[model]
        task_type = TASK_MAP.get(task, task) if task and task in TASK_MAP else task

        job_id, result = start_training(
            dataset_url, model_name, task_type, hyperparams, api_token,
            text_column, label_column, train_split, fp16, push_to_hub, hub_repo_id, seed
        )

        monitor_training(job_id, api_token)

        with open(output, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to {output}")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise click.Abort()

if __name__ == '__main__':
    train()