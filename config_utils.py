import json
from pathlib import Path
from typing import Optional

def load_api_token(api_token: Optional[str] = None) -> Optional[str]:
    """Load API token from argument or config file."""
    if api_token:
        return api_token
    config_path = Path.home() / '.easytrain' / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config.get('api_token')
    return None

def save_api_token(api_token: str) -> None:
    """Save API token to config file."""
    config_dir = Path.home() / '.easytrain'
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump({'api_token': api_token}, f)