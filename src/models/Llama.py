import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .Model import Model


class Llama(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.max_output_tokens = config["params"]["max_output_tokens"]

        api_pos = int(config["api_key_info"]["api_key_use"])
        hf_token = config["api_key_info"]["api_keys"][api_pos]

        # Use AutoTokenizer for better compatibility with Llama2 and Llama3 models
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, token=hf_token)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name, 
            torch_dtype=torch.float16, 
            token=hf_token,
            device_map="auto"
        )
        
        # Check if this is a Llama3 Instruct model
        self.is_llama3_instruct = "llama-3" in self.name.lower() and "instruct" in self.name.lower()

    def query(self, msg):
        if self.is_llama3_instruct:
            # Use chat template for Llama3 Instruct models
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Provide short and concise answers."},
                {"role": "user", "content": msg}
            ]
            input_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        else:
            input_ids = self.tokenizer(msg, return_tensors="pt").input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_output_tokens,
                do_sample=self.temperature > 0,
                **({"temperature": self.temperature} if self.temperature > 0 else {}),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        result = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        return result.strip()