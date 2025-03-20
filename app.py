from vllm import LLM
from vllm.sampling_params import SamplingParams
os.environ["VLLM_USE_V1"] = "0"

class InferlessPythonModel:
    def initialize(self):
        model_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"  # Specify the model repository ID
        # Define sampling parameters for model generation
        self.sampling_params = SamplingParams(max_tokens=512, temperature=0.15)
        # Initialize the LLM object
        self.llm = LLM(model=model_id)
        
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
        prompts = inputs["prompt"]  
        outputs = llm.chat(messages, sampling_params=sampling_params)

        result_output = outputs[0].outputs[0].text

        # Return a dictionary containing the result
        return {'generated_text': result_output}

    def finalize(self):
        pass
