"""
HuggingFace LLaMA-3-8B-Instruct integration with 4-bit quantization
This module provides a wrapper for using Meta's LLaMA-3-8B-Instruct model
with memory-efficient 4-bit quantization for Kaggle environments.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.models.Model import Model


class HuggingFaceLLaMA(Model):
    """
    HuggingFace LLaMA-3-8B-Instruct with 4-bit quantization
    Designed for Kaggle GPU environments (T4/P100)
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        
        # Get HuggingFace token if provided
        api_pos = int(config["api_key_info"]["api_key_use"])
        hf_token = config["api_key_info"]["api_keys"][api_pos]
        
        # Use token if it's a valid string (not placeholder)
        use_auth_token = hf_token if hf_token and "your" not in hf_token.lower() else None
        
        # 4-bit quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        print(f"Loading {self.name} with 4-bit quantization...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.name,
            token=use_auth_token,
            trust_remote_code=True
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with 4-bit quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            quantization_config=bnb_config,
            device_map="auto",
            token=use_auth_token,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        print(f"Model loaded successfully on {self.device}")
        
    def query(self, msg):
        """
        Generate response for a given prompt
        
        Args:
            msg (str): Input prompt text
            
        Returns:
            str: Generated response
        """
        # Prepare input
        inputs = self.tokenizer(msg, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_output_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and extract only the new tokens (not the input prompt)
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the response
        response = full_response[len(msg):].strip()
        
        return response
