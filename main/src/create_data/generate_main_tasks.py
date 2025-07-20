import openai
from openai import OpenAI
import json
from utils import *
import re
import os
import random

root_write_folder = "/public/home/hmqiu/xzhang/task_planning/main/data/train_main_task_3000"

test_gold_main_tasks_folder = "/public/home/hmqiu/xzhang/task_planning/main/data/new_main_task"


def read_json_file(filename):
    if filename.endswith(".json"):
        with open(filename, 'r')as f:
            all_data = json.load(f)
            return all_data
    elif filename.endswith(".jsonl"):
        all_data = []
        with open(filename, 'r')as f:
            for line in f:
                data = json.loads(line)
                all_data.append(data)
        return all_data
    else:
        raise "your filename is wrong"

def write_data(data, filename):
    with open(filename, 'w')as f:
        json.dump(data, f, indent=2)

def process_response(raw_response: str):
    response = re.findall(r"\[.*\]", raw_response, re.DOTALL)[0]
    main_tasks = json.loads(response)
    return main_tasks


def get_gold_main_tasks(candidate_categories):
    gold_category_2_tasks = {}
    all_kinds = os.listdir(test_gold_main_tasks_folder)
    for now_kind in all_kinds:
        kind_name = now_kind[:-5]
        now_filename = os.path.join(test_gold_main_tasks_folder, now_kind)
        now_kind_tasks = read_file(now_filename)
        random.shuffle(now_kind_tasks)
        if len(now_kind_tasks) < 10:
            gold_category_2_tasks[kind_name] = now_kind_tasks
        else:
            gold_category_2_tasks[kind_name] = now_kind_tasks[:10]

    return gold_category_2_tasks

        

def use_gpt_generate_data(client, raw_model_name:str="gpt-3.5-turbo"):

    
    all_candidate_categories = ["Home Improvement and DIY Projects", "Business and Financial Activities", "Writing and Research", "Cooking and Food Preparation", "Arts and Crafts", "Health and Fitness", "Gardening and Outdoor Activities", "Technology and Digital Management", "Personal Care and Household Management", "Educational and Skill Development"]
    all_candidate_categories_filename = ["Home_Improvement_and_DIY_Projects", "Business_and_Financial_Activities", "Writing_and_Research", "Cooking_and_Food_Preparation", "Arts_and_Crafts", "Health_and_Fitness", "Gardening_and_Activities", "Technology_and_Digital_Management", "Personal_Care_and_Household_Management", "Educational_and_Skill_Development"]

    gold_category_2_tasks = get_gold_main_tasks(all_candidate_categories_filename)

    all_gold_test_main_tasks = read_data_from_folder(test_gold_main_tasks_folder)

    # 每个类别生成多少个任务 可以按情况修改
    # all_candidate_number = [200, 100, 100, 200, 200, 20, 100, 100, 60, 20]
    all_candidate_number = [50, 0, 0, 100, 50, 0, 50, 0, 0, 0]
    for i in range(len(all_candidate_categories_filename)):
        now_category = all_candidate_categories[i]
        now_category_filename = all_candidate_categories_filename[i]

        gold_candidate_tasks = gold_category_2_tasks[now_category_filename]

        category_number = all_candidate_number[i]
        if category_number == 0:
            continue
        # now_example = data_example[i]
        
        system_content = f'''In reality, there are many complex tasks that need to be completed by combining multiple subtasks, such as building a simple cabin, organizing a camping trip, etc. Now I need you to generate {category_number} tasks in the category of {now_category}. You need to return the answer in JSON format. The return format is '[\"task1\",\"task2\", ...].
for example: {str(gold_candidate_tasks)}'''
        
        messages = [{"role": "system", "content": system_content}]
        
        completion = client.chat.completions.create(
                model=model_name,
                messages=messages
        )
        response = completion.choices[0].message.content
        main_tasks = process_response(response)
        category_write_filename = os.path.join(root_write_folder, now_category_filename+".json")
        has_generate_data = []
        try:
            has_generate_data = read_file(category_write_filename)
            for data in main_tasks:
                if data not in has_generate_data and data not in all_gold_test_main_tasks:
                    has_generate_data.append(data)
        except:
            has_generate_data = main_tasks
            print("Generate data from scratch")

        write_data(has_generate_data, category_write_filename)
        

    

if __name__ == "__main__":
    model_name = "gpt-4o-2024-08-06"
    # model_name = "gpt-3.5-turbo"
    client = OpenAI(api_key="sk-C9Gzkape7lwmtKS2B2Ef4b2d3a7a4c3bB7A641AfB24397E4", base_url="https://www.gptapi.us/v1")
    use_gpt_generate_data(client, model_name)

    all_foldername = "/public/home/hmqiu/xzhang/task_planning/main/data/train_main_task_3000"
    all_now_main_tasks = read_data_from_folder(all_foldername)
    print(len(all_now_main_tasks))





