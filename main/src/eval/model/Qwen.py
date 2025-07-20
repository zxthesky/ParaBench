from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig
from model.generate_configs import GenerateConfigs
import transformers
import torch

class QwenModel:
    
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
        prompts: dict,
        template: str = None,
        generate_configs: GenerateConfigs=None,
        history: list = None,
    ) -> list:
        template = self.template
        if generate_configs == None:
            generate_params = GenerateConfigs()
        params = self.generate_params(generate_params)

        if template == "default":
            
            messages = [
                {"role": "system", "content": prompts["instruction"]},
                {"role": "user", "content": prompts["input"]}
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(text, return_tensors="pt")
            inputs["input_ids"] = inputs["input_ids"].cuda()

            # inputs.update(params)
            
            output = self.model.generate(**inputs, max_new_tokens=256)
            #predict = self.tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            predict = self.tokenizer.decode(output[0].tolist())[len(text):]
            final_predict = predict.replace("<|im_end|>", "")
            return final_predict

        else:
            output, _ = self.model.chat(self.tokenizer, prompts, history=history, **params)
            return output



    def generate_params(
        self,
        generate_configs: GenerateConfigs,
    ):
        kargs = generate_configs.dict()
        params = {
            "max_new_tokens": kargs.get("max_new_tokens", 128),
            "top_k": kargs.get("top_k", 50),
            "top_p": kargs.get("top_p", 0.95),
            "temperature": kargs.get("temperature", 1.0),
        }



        return params
    
    def generate_stream(self, params):

        prompt = params["prompt"]
        len_prompt = len(prompt)
        temperature = float(params.get("temperature", 1.0))
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = int(params.get("top_k", -1))  # -1 means disable
        max_new_tokens = int(params.get("max_new_tokens", 256))
        stop_str = params.get("stop", None)
        echo = bool(params.get("echo", True))
        stop_token_ids = params.get("stop_token_ids", [])
        

        input_ids = self.tokenizer(prompt).input_ids
        input_echo_len = len(input_ids)
        output_ids = list(input_ids)


        past_key_values = out = None

        for i in range(max_new_tokens):
            if i == 0:
                out = self.model(
                        input_ids=torch.as_tensor([input_ids], device="cuda"),
                        use_cache=True,
                    )
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = self.model(
                        input_ids=torch.as_tensor([input_ids], device="cuda"),
                        use_cache=True,
                        past_key_values=out.past_key_values,
                    )
                logits = out.logits
                past_key_values = out.past_key_values

            




    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path,  low_cpu_mem_usage=True, device_map="auto", trust_remote_code=self.trust_remote_code).eval()


if __name__ == '__main__':

    # qwen_model_parameter_path = r"../../ToolBench-master/model_parameter/Qwen1.5-1.8B"
    qwen_model_parameter_path = r"./model_parameter/Qwen2_7b_instruct"
    print(qwen_model_parameter_path)
    Qwen_model = QwenModel(qwen_model_parameter_path)
    generate_params = GenerateConfigs()
    
    response = Qwen_model.generate("Can you please modify my appointment scheduled for March 25th with Dr. Kim to March 26th with Dr. Lee?\nAI: Sure, I can help you with that. Please provide me with the appointment ID and the new appointment date and doctor's name.\nUser: The appointment ID is 34567890 and the new date is March 26th with Dr. Lee.\nAI: Alright. I'll modify your appointment now.\nAPI-Request: [ModifyRegistration(appointment_id='34567890', new_appointment_date='2023-03-26', new_appointment_doctor='Dr. Lee')]->success\nAI: Your appointment has been modified successfully.\nUser: Can you also check my health data for the past week?\nAI: Of course. To do that, I'll need your user ID and the start and end time of the time span you want to check.\nUser: My user ID is J46801 and I want to check from March 5th to March 12th.\nAI: Got it.\nGenerate API Request:\n", generate_configs = generate_params)
    print('---------------------')
    print(response)



