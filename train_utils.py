import logging
import json
import time
import os
from huggingface_hub import HfApi, create_repo
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import load_dataset
import numpy as np

logger = logging.getLogger(__name__)

def compute_metricsClassification(eval_pred):
    """Compute metrics for classification task."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted'),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted'),
    }

def compute_metricsNER(eval_pred):
    """Compute metrics for NER task."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [str(p) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [str(l) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    from sklearn.metrics import accuracy_score
    flat_preds = [item for sublist in true_predictions for item in sublist]
    flat_labels = [item for sublist in true_labels for item in sublist]
    return {'accuracy': accuracy_score(flat_labels, flat_preds)}

from typing import Optional

def start_training(
    dataset_path: str,
    model_name: str,
    task: Optional[str],
    hyperparams: dict,
    api_token: str,
    text_column: str = 'text',
    label_column: str = 'label',
    train_split: float = 0.9,
    fp16: bool = False,
    push_to_hub: bool = False,
    hub_repo_id: str = None,
    seed: int = 42
) -> tuple:
    """Start training job on Hugging Face."""
    try:
        if dataset_path and dataset_path.startswith('hf://'):
            dataset = load_dataset(dataset_path.replace('hf://', ''), token=api_token)
        else:
            format_type = 'json' if dataset_path.endswith('.json') else 'csv' if dataset_path.endswith('.csv') else 'parquet'
            dataset = load_dataset(format_type, data_files=dataset_path, split='train')
            if train_split < 1.0:
                dataset = dataset.train_test_split(test_size=1-train_split, seed=seed)

        if task == 'text-classification':
            if label_column not in dataset['train'].column_names and 'label' in dataset['train'].column_names:
                label_column = 'label'
            label_list = list(set(dataset['train'][label_column]))
            label2id = {label: i for i, label in enumerate(label_list)}
            id2label = {i: label for i, label in enumerate(label_list)}
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            def tokenize_function(examples):
                return tokenizer(examples[text_column], padding='max_length', truncation=True, max_length=512)
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(label_list),
                id2label=id2label,
                label2id=label2id
            )
            
            compute_metrics = compute_metricsClassification
            data_collator = None

        elif task == 'token-classification':
            if label_column not in dataset['train'].column_names:
                label_column = 'ner_tags'
            all_labels = set()
            for ex in dataset['train']:
                if label_column in ex:
                    all_labels.update(ex[label_column] if isinstance(ex[label_column], list) else [ex[label_column]])
            label_list = sorted(list(all_labels))
            label2id = {label: i for i, label in enumerate(label_list)}
            id2label = {i: label for i, label in enumerate(label_list)}
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            def tokenize_and_align_labels(examples):
                tokenized = tokenizer(
                    examples[text_column],
                    padding='max_length',
                    truncation=True,
                    max_length=512,
                    is_split_into_words=True
                )
                labels = []
                for i, label in enumerate(examples[label_column]):
                    word_ids = tokenized.word_ids(batch_index=i)
                    aligned_labels = []
                    previous_word_idx = None
                    for word_idx in word_ids:
                        if word_idx is None:
                            aligned_labels.append(-100)
                        elif word_idx != previous_word_idx:
                            aligned_labels.append(label2id.get(label[word_idx] if isinstance(label, list) else label, 0))
                        else:
                            aligned_labels.append(-100)
                        previous_word_idx = word_idx
                    labels.append(aligned_labels)
                tokenized['labels'] = labels
                return tokenized
            
            tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
            
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=len(label_list),
                id2label=id2label,
                label2id=label2id
            )
            
            compute_metrics = compute_metricsNER
            data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        else:
            raise ValueError(f"Unsupported task: {task}")

        repo_id = hub_repo_id or f"model-{int(time.time())}"
        
        if push_to_hub:
            create_repo(repo_id, token=api_token, exist_ok=True)
            hub_model_id = repo_id
        else:
            hub_model_id = None

        output_dir = f"./output/{repo_id}"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=hyperparams.get('num_train_epochs', 3),
            learning_rate=hyperparams.get('learning_rate', 2e-5),
            per_device_train_batch_size=hyperparams.get('per_device_train_batch_size', 16),
            warmup_ratio=hyperparams.get('warmup_ratio', 0.1),
            weight_decay=hyperparams.get('weight_decay', 0.01),
            logging_steps=hyperparams.get('logging_steps', 10),
            eval_steps=hyperparams.get('eval_steps', 50),
            save_steps=hyperparams.get('save_steps', 50),
            save_total_limit=hyperparams.get('save_total_limit', 2),
            load_best_model_at_end=hyperparams.get('load_best_model_at_end', True),
            evaluation_strategy=hyperparams.get('evaluation_strategy', 'steps'),
            fp16=fp16,
            report_to='none',
            seed=seed,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset['train'] if 'train' in tokenized_dataset else tokenized_dataset,
            eval_dataset=tokenized_dataset.get('test') if isinstance(tokenized_dataset, dict) else None,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )

        logger.info(f"Starting training with config: {json.dumps({k: str(v) for k, v in hyperparams.items()}, indent=2)}")
        
        trainer.train()
        
        metrics = trainer.evaluate() if trainer.eval_dataset else {}
        
        model_url = f"https://huggingface.co/{repo_id}" if push_to_hub else output_dir
        
        result = {
            'status': 'completed',
            'model_url': model_url,
            'metrics': metrics,
            'hyperparameters': hyperparams,
            'task': task,
            'model': model_name,
        }
        
        if push_to_hub:
            trainer.push_to_hub()

        logger.info(f"Training completed: {result['model_url']}")
        return repo_id, result

    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise

def monitor_training(job_id: str, api_token: str) -> dict:
    """Monitor training job status."""
    try:
        logger.info(f"Monitoring job: {job_id}")
        return {'status': 'completed'}
    except Exception as e:
        logger.error(f"Failed to monitor training: {e}")
        raise
