"""
对数据进行翻译，中文更方便检查
"""
from utils import read_file, write_file
from zhipuai import ZhipuAI
from copy import deepcopy
import re
import json

system_content = """你是一个外语专家，你非常擅长将英语转化为中文。现在我给你数组格式的内容，你需要将它转化为中文，并按照原来格式返回，下面我给出需要翻译的内容，答案以数组的形式呈现，你需要以JSON格式返回。
下面是例子：
user: ['gather ingredients', 'prepare broth', 'chop onions', 'chop mushrooms', 'grate cheese', 'heat oil in pan', 'sauté onions', 'add mushrooms to pan', 'add rice to pan', 'add broth gradually', 'stir continuously', 'add cheese', 'season with salt and pepper', 'serve risotto', 'Make creamy mushroom risotto']'
Answer: ["准备食材", "准备高汤", "切洋葱", "切蘑菇", "刨奶酪", "在平底锅中加热油", "炒洋葱", "将蘑菇加入锅中", "将米饭加入锅中", "逐渐加入高汤", "不断搅拌", "加入奶酪", "用盐和胡椒调味", "盛出烩饭", "制作奶油蘑菇烩饭"]"""


def get_chainese_dependencies(dependencies, cha_subtasks, subtasks):
    subtasks_dict = {}
    for i in range(len(subtasks)):
        subtasks_dict[subtasks[i].lower()] = str(i) + " " + cha_subtasks[i]

    cha_dependencies = []
    for edge in dependencies:
        temp_data = {}
        cha_subtask1 = subtasks_dict[edge["subtask1"].lower()]
        cha_subtask2 = subtasks_dict[edge["subtask2"].lower()]
        temp_data["subtask1"] = cha_subtask1
        temp_data["subtask2"] = cha_subtask2
        cha_dependencies.append(temp_data)
    return cha_dependencies

def delete_has_translated_data(all_data_need_translate, has_translated_tasks):
    need_translate_data = []
    for data in all_data_need_translate:
        main_task = data["task"]
        if main_task not in has_translated_tasks:
            need_translate_data.append(data)

    return need_translate_data
        

def main(filename, write_filename):
    try:
        has_translate_data = read_file(write_filename)
    except:
        has_translate_data = []
    has_translate_main_tasks = []
    for data in has_translate_data:
        now_task = data["task"]
        has_translate_main_tasks.append(now_task)

    client = ZhipuAI(api_key="4dbb1b507ca4ac33a60c20e4280027a7.GsY3PtMEHNYDY8Eq")
    model = "GLM-4-Flash"
    all_data = read_file(filename)
    all_data = delete_has_translated_data(all_data, has_translate_main_tasks)
    final_data = []
    print(len(all_data))
    now_number = 0
    for data in all_data:
        temp_data = {}
        for key in data:
            temp_data[key] = deepcopy(data[key])
        subtasks = data["substeps"]
        main_task = data["task"]
        raw_subtasks = [i["step"] for i in subtasks]
        raw_subtasks.append(main_task)
        dependencies = data["dependencies"]
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": str(raw_subtasks)}
        ]
        for i in range(10):
            try:
                response = client.chat.completions.create(
                    model = model,
                    messages = messages
                )
                raw_response = response.choices[0].message.content

                
                translate_data = re.findall(r"\[.*\]", raw_response, re.DOTALL)[0]
                
                cha_subtasks = json.loads(translate_data)
                assert len(cha_subtasks) == len(raw_subtasks)

                temp_data["cha_dependencies"] = get_chainese_dependencies(dependencies, cha_subtasks, raw_subtasks)
                print(f"-------  {now_number}  -----------")
                final_data.append(temp_data)
            except:
                continue
            break
        now_number += 1

        
    
    write_file(final_data, write_filename)




if __name__ == "__main__":
    filename = "/data/xzhang/task_planning/main/data/train_data/all_data_use_category.json"
    write_filename = "/data/xzhang/task_planning/main/process_data/processed_data/train_data_category.json"
    main(filename, write_filename)
