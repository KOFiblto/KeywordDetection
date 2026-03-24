import os
import json

def load_config():
    """Reads the shared config.json from the repository root."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, ".."))
    config_path = os.path.join(root_dir, "config.json")
    
    if not os.path.exists(config_path):
        # Fallback to current directory for edge cases
        config_path = "config.json"
        
    with open(config_path, "r") as f:
        return json.load(f)

def get_keywords():
    return load_config()["keywords"]

def get_config_value(key, default=None):
    return load_config().get(key, default)
