import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_utils import load_api_token, save_api_token
from cli import MODEL_MAP, TASK_MAP, DEFAULT_HYPERPARAMS


class TestConfigUtils:
    def test_load_api_token_from_argument(self):
        token = load_api_token(api_token="test_token_123")
        assert token == "test_token_123"

    @patch('config_utils.Path.home')
    def test_load_api_token_from_config(self, mock_home):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_home.return_value = Path(tmpdir)
            config_dir = Path(tmpdir) / '.easytrain'
            config_dir.mkdir()
            config_path = config_dir / 'config.json'
            with open(config_path, 'w') as f:
                json.dump({'api_token': 'config_token'}, f)
            
            token = load_api_token()
            assert token == 'config_token'

    def test_load_api_token_returns_none(self):
        with patch('config_utils.Path.home') as mock_home:
            with tempfile.TemporaryDirectory() as tmpdir:
                mock_home.return_value = Path(tmpdir)
                token = load_api_token()
                assert token is None

    @patch('config_utils.Path.home')
    def test_save_api_token(self, mock_home):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_home.return_value = Path(tmpdir)
            save_api_token("new_token")
            config_path = Path(tmpdir) / '.easytrain' / 'config.json'
            assert config_path.exists()
            with open(config_path) as f:
                config = json.load(f)
                assert config['api_token'] == 'new_token'


class TestCliConstants:
    def test_model_map_contains_expected_models(self):
        assert 'bert' in MODEL_MAP
        assert 'distilbert' in MODEL_MAP
        assert 'roberta' in MODEL_MAP
        assert 'albert' in MODEL_MAP
        assert 'electra' in MODEL_MAP
        assert 'deberta' in MODEL_MAP

    def test_task_map_contains_expected_tasks(self):
        assert 'classification' in TASK_MAP
        assert 'ner' in TASK_MAP
        assert 'sentiment' in TASK_MAP
        assert 'qa' in TASK_MAP

    def test_default_hyperparams_has_required_keys(self):
        required_keys = [
            'learning_rate', 'num_train_epochs', 'per_device_train_batch_size',
            'warmup_ratio', 'weight_decay', 'logging_steps', 'eval_steps',
            'save_steps', 'save_total_limit', 'load_best_model_at_end',
            'evaluation_strategy'
        ]
        for key in required_keys:
            assert key in DEFAULT_HYPERPARAMS

    def test_default_hyperparams_values(self):
        assert DEFAULT_HYPERPARAMS['learning_rate'] == 2e-5
        assert DEFAULT_HYPERPARAMS['num_train_epochs'] == 3
        assert DEFAULT_HYPERPARAMS['per_device_train_batch_size'] == 16
        assert DEFAULT_HYPERPARAMS['warmup_ratio'] == 0.1
        assert DEFAULT_HYPERPARAMS['weight_decay'] == 0.01
        assert DEFAULT_HYPERPARAMS['logging_steps'] == 10
        assert DEFAULT_HYPERPARAMS['eval_steps'] == 50


class TestDatasetUtils:
    @patch('dataset_utils.load_dataset')
    def test_validate_dataset_success(self, mock_load):
        mock_dataset = MagicMock()
        mock_dataset['train'].column_names = ['text', 'label']
        mock_load.return_value = mock_dataset
        
        from dataset_utils import validate_dataset
        result = validate_dataset('test.csv', 'csv', 'text', 'label')
        assert result is True

    @patch('dataset_utils.load_dataset')
    def test_validate_dataset_missing_text_column(self, mock_load):
        mock_dataset = MagicMock()
        mock_dataset['train'].column_names = ['label']
        mock_load.return_value = mock_dataset
        
        from dataset_utils import validate_dataset
        result = validate_dataset('test.csv', 'csv', 'text', 'label')
        assert result is False

    @patch('dataset_utils.load_dataset')
    def test_validate_dataset_optional_label(self, mock_load):
        mock_dataset = MagicMock()
        mock_dataset['train'].column_names = ['text']
        mock_load.return_value = mock_dataset
        
        from dataset_utils import validate_dataset
        result = validate_dataset('test.csv', 'csv', 'text')
        assert result is True

    @patch('dataset_utils.load_dataset')
    def test_validate_dataset_exception(self, mock_load):
        mock_load.side_effect = Exception("Invalid file")
        
        from dataset_utils import validate_dataset
        result = validate_dataset('test.csv', 'csv', 'text')
        assert result is False
