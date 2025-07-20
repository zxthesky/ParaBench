import transformers
from utils import read_file
import os
import dashscope
from prompt import *
import json
from tqdm import *
import re

my_qwen_api_key = "sk-1a88212a8edb42f6aa85bde8b7c995d5"

def read_folder_data(folder_path):
    all_tasks = []
    all_filename = os.listdir(folder_path)
    for filename in all_filename:
        now_file_path = os.path.join(folder_path, filename)
        now_tasks = read_file(now_file_path)
        all_tasks.extend(now_tasks)
    return all_tasks

def write_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    task_folder = "/data/xzhang/task_planning/main/data/main_tasks"

    dependencies_data_filename = "/data/xzhang/task_planning/main/create_data/batch_process_data/use_small_model/dependencies_data/dependencies.json"
    unrelevant_data_filename = "/data/xzhang/task_planning/main/create_data/batch_process_data/use_small_model/unrelevants_data/unrelevants.json"

    all_tasks = read_folder_data(task_folder)
    all_tasks = all_tasks

    dependency_system_prompt = system_prompt_has_dependeny
    unrelevant_system_prompt = system_prompt_no_relevant
    dependencies_datas = []
    unrelevant_datas = []

    for i in tqdm(range(len(all_tasks))):
        task = all_tasks[i]
        now_dependencies = {"task": task, "data": []}
        now_unrelevants = {"task": task, "data": []}
        now_dependency_system_prompt = dependency_system_prompt
        now_unrelevant_system_prompt = unrelevant_system_prompt

        dependency_messages = [
            {'role': 'system', 'content': now_dependency_system_prompt},
            {'role': 'user', 'content': task}
        ]
        dependency_response = dashscope.Generation.call(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=my_qwen_api_key,
            model="qwen-plus-latest", # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=dependency_messages,
            result_format='message'
            )["output"]["choices"][0]["message"]["content"]
        
        unrelevant_messages = [
            {'role': 'system', 'content': now_unrelevant_system_prompt},
            {'role': 'user', 'content': task}
        ]
        unrelevant_response = dashscope.Generation.call(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=my_qwen_api_key,
            model="qwen-plus-latest", # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=dependency_messages,
            result_format='message'
            )["output"]["choices"][0]["message"]["content"]
        
        try:
            dependency_re = re.findall(r"\[.*\{.*\}.*\]", dependency_response, re.DOTALL)[0]
            unrelevant_re = re.findall(r"\[.*\{.*\}.*\]", unrelevant_response, re.DOTALL)[0]

            final_dependencies = json.loads(dependency_re)
            final_unrelevant = json.loads(unrelevant_re)
            now_dependencies["data"] = final_dependencies
            now_unrelevants["data"] = final_unrelevant

            dependencies_datas.append(now_dependencies)
            unrelevant_datas.append(now_unrelevants)
            write_file(dependencies_datas, dependencies_data_filename)
            write_file(unrelevant_datas, unrelevant_data_filename)
        except:
            continue

    write_file(dependencies_datas, dependencies_data_filename)
    write_file(unrelevant_datas, unrelevant_data_filename)






    


