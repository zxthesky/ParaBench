from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig

import transformers
import torch


def change_prompt(prompt, gold_answer):
    raw_split_prompt = prompt.split('\n{\"name\"')
    final_prompt = raw_split_prompt[0]
    gold_answer_api_name = ""
    
    for i in gold_answer[14:]:
        if i != '(':
            gold_answer_api_name += i
        else:
            break
    # print(gold_answer)
    # print(gold_answer_api_name)
    for i in range(1, len(raw_split_prompt)):
        if gold_answer_api_name in raw_split_prompt[i]:
            final_prompt += '\n{\"name\"' + raw_split_prompt[i]
            break
    return final_prompt


def get_prompt_seal_tools(i):
    raw_prompt = i[0]

class LLAMA3Model:
    
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
        messages: list,
    ) -> list:
        params = {
            'max_new_tokens': 1024,
            'beam_bums': 1,
            'top_k': 50,
            'top_p': 0.95,
            'temperature': 1.0,
            'length_penalty': 1.0,
            'presence_penalty': 1.0,
            'stop_words': [],
            'template': "default"
        }
        
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




    def generate_params(
        self,
        generate_configs
    ):
        kargs = generate_configs.dict()
        params = {
            "max_new_tokens": kargs.get("max_new_tokens", 128),
            "top_k": kargs.get("top_k", 50),
            "top_p": kargs.get("top_p", 0.95),
            "temperature": kargs.get("temperature", 1.0),
        }



        return params

            




    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path,  low_cpu_mem_usage=True, device_map="auto", trust_remote_code=self.trust_remote_code).eval()


if __name__ == '__main__':

    # qwen_model_parameter_path = r"../../ToolBench-master/model_parameter/Qwen1.5-1.8B"
    qwen_model_parameter_path = r"./model_parameter/llama3-8b"
    print(qwen_model_parameter_path)
    Qwen_model = LLAMA3Model(qwen_model_parameter_path)

    
    response = Qwen_model.generate("Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the previous dialogue context.\nThe current year is 2023.\nInput: \nUser: User's utterence\nAI: AI's response\n\nExpected output:\nAPI-Request: [ApiName(key1='value1', key2='value2', ...)]\n\nAPI descriptions:\n{\"name\": \"QueryHealthData\", \"description\": \"This API queries the recorded health data in database of a given user and time span.\", \"input_parameters\": {\"user_id\": {\"type\": \"str\", \"description\": \"The user id of the given user. Cases are ignored.\"}, \"start_time\": {\"type\": \"str\", \"description\": \"The start time of the time span. Format: %Y-%m-%d %H:%M:%S\"}, \"end_time\": {\"type\": \"str\", \"description\": \"The end time of the time span. Format: %Y-%m-%d %H:%M:%S\"}}, \"output_parameters\": {\"health_data\": {\"type\": \"list\", \"description\": \"The health data of the given user and time span.\"}}}\n{\"name\": \"CancelRegistration\", \"description\": \"This API cancels the registration of a patient given appointment ID.\", \"input_parameters\": {\"appointment_id\": {\"type\": \"str\", \"description\": \"The ID of appointment.\"}}, \"output_parameters\": {\"status\": {\"type\": \"str\", \"description\": \"The status of cancellation.\"}}}\n{\"name\": \"ModifyRegistration\", \"description\": \"This API modifies the registration of a patient given appointment ID.\", \"input_parameters\": {\"appointment_id\": {\"type\": \"str\", \"description\": \"The ID of appointment.\"}, \"new_appointment_date\": {\"type\": \"str\", \"description\": \"The new appointment date. Format: %Y-%m-%d.\"}, \"new_appointment_doctor\": {\"type\": \"str\", \"description\": \"The new appointment doctor.\"}}, \"output_parameters\": {\"status\": {\"type\": \"str\", \"description\": \"The status of modification.\"\nUser: Can you please modify my appointment scheduled for March 25th with Dr. Kim to March 26th with Dr. Lee?\nAI: Sure, I can help you with that. Please provide me with the appointment ID and the new appointment date and doctor's name.\nUser: The appointment ID is 34567890 and the new date is March 26th with Dr. Lee.\nAI: Alright. I'll modify your appointment now.\nAPI-Request: [ModifyRegistration(appointment_id='34567890', new_appointment_date='2023-03-26', new_appointment_doctor='Dr. Lee')]->success\nAI: Your appointment has been modified successfully.\nUser: Can you also check my health data for the past week?\nAI: Of course. To do that, I'll need your user ID and the start and end time of the time span you want to check.\nUser: My user ID is J46801 and I want to check from March 5th to March 12th.\nAI: Got it.\nGenerate API Request:\n")
    print('---------------------')
    print(response)
    print('----------------------------------------------')



