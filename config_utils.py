import json
from pathlib import Path

def load_api_token(api_token: str = None) -> str:
    """Load API token from argument or config file."""
    if api_token:
        return api_token
    config_path = Path.home() / '.easytrain' / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config.get('api_token')
    return None