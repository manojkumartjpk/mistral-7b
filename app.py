import os
import torch
import json
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration, BitsAndBytesConfig

os.environ["VLLM_USE_V1"] = "0"

class InferlessPythonModel:
    def initialize(self):
        model_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"  # Specify the model repository ID
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="cuda",
        )
        
    def infer(self,inputs):
        SYSTEM_PROMPT = "You are a conversational agent that always answers straight to the point, always end your accurate response with an ASCII drawing of a cat."
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": inputs["prompt"]
            },
        ]

        inputs = self.tokenizer(json.dumps(messages), return_tensors="pt").to("cuda")
        
        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(generated_text)
        return {'generated_text': generated_text}

    def finalize(self):
        pass
