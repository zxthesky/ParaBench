from transformers import AutoModelForCausalLM, AutoTokenizer

class  LLAMA3Model:   
    def __init__(self, model_path: str, peft_path: str = None, template: str = "default", trust_remote_code=True, tensor_parallel_size=1, gpu_memory_utilization=0.25):
        self.model_path = model_path
        self.peft_path = peft_path
        self.template  = template
        self.trust_remote_code = trust_remote_code
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.load_model()
        self.history = []

    def generate(
        self,
        system_content: str,
    ):
        
        messages = [{"role": "system","content": system_content}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs["input_ids"] = inputs["input_ids"].cuda()           
        output = self.model.generate(**inputs, max_new_tokens=256)
        predict = self.tokenizer.decode(output[0].tolist())[len(text):]
        final_predict = predict.replace("<|im_end|>", "")
        return final_predict
    
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path,  low_cpu_mem_usage=True, device_map="auto", trust_remote_code=self.trust_remote_code).eval()



