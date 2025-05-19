import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)

def validate_dataset(dataset_path: str, file_format: str, text_column: str, label_column: str = None) -> bool:
    """Validate dataset format and required columns."""
    try:
        dataset = load_dataset(file_format, data_files=dataset_path)
        columns = dataset['train'].column_names
        if text_column not in columns:
            logger.error(f"Dataset must have a '{text_column}' column")
            return False
        if label_column and label_column not in columns:
            logger.error(f"Dataset must have a '{label_column}' column")
            return False
        return True
    except Exception as e:
        logger.error(f"Invalid dataset: {e}")
        return False

def upload_dataset(dataset_path: str, repo_name: str, api_token: str, file_format: str) -> str:
    """Upload dataset to Hugging Face Datasets."""
    try:
        logger.info(f"Uploading dataset: {dataset_path}")
        dataset = load_dataset(file_format, data_files=dataset_path)
        dataset.push_to_hub(repo_name, token=api_token)
        dataset_url = f"{repo_name}"
        logger.info(f"Dataset uploaded to {dataset_url}")
        return dataset_url
    except Exception as e:
        logger.error(f"Failed to upload dataset: {e}")
        raise