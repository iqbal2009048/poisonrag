from .PaLM2 import PaLM2
from .Vicuna import Vicuna
from .GPT import GPT
from .Llama import Llama
import json
import sys
import os

# Add parent directory to path to import HuggingFaceLLaMA
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

def create_model(config_path):
    """
    Factory method to create a LLM instance
    """
    config = load_json(config_path)

    provider = config["model_info"]["provider"].lower()
    if provider == 'palm2':
        model = PaLM2(config)
    elif provider == 'vicuna':
        model = Vicuna(config)
    elif provider == 'gpt':
        model = GPT(config)
    elif provider == 'llama':
        model = Llama(config)
    elif provider == 'huggingface' or provider == 'huggingface_llama':
        # Import HuggingFaceLLaMA dynamically to avoid dependency issues
        from llm_huggingface import HuggingFaceLLaMA
        model = HuggingFaceLLaMA(config)
    else:
        raise ValueError(f"ERROR: Unknown provider {provider}")
    return model