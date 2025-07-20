from model import LLAMA3Model
from utils import read_file, write_file
import re
from prompt import *
from tqdm import  *
model_path = "/data/xzhang/model_parameter/LLama3-instruct"

def process_answer(answer):
    judge = re.findall("\[.*\]", answer)
    if judge != None and len(judge) != 0:
        judge = judge[0]
        return judge.lower()
    else:
        return False


def generate_answer(model, data, model_type:str="local"):
    if model_type == "local":
        response = model.generate(data)
    
    return response



if __name__ == "__main__":
    detect_data_filepath = "/data/xzhang/task_planning/main/data/gpt-4o-2024-08-06/all_data.json"
    all_data = read_file(detect_data_filepath)
    model_type = "local"
    raw_system_prompt = judge_prompt
    model = LLAMA3Model(model_path)

    gold_data_write_filepath = "/data/xzhang/task_planning/main/create_data/data/gold_data.json"
    review_data_write_filepath = "/data/xzhang/task_planning/main/create_data/data/review_data.json"
    delete_data_write_filepath = "/data/xzhang/task_planning/main/create_data/data/detect_data.json"

    gold_threshold = 0.9
    gold_data = []
    review_threshold = 0.7
    review_data = []
    delete_data = []
    for i in tqdm(range(len(all_data))):
        data = all_data[i]
        main_task = data["task"]
        dependencies = data["dependencies"]
        this_data_scores = 0
        all_edge_number = 0
        for edge in dependencies:
            subtask1 = edge["subtask1"]
            subtask2 = edge["subtask2"]
            if subtask1.lower() != main_task.lower():
                system_content = judge_prompt.replace("(main_task)", main_task)
                system_content = system_content.replace("(subtask1)", subtask1)
                system_content = system_content.replace("(subtask2)", subtask2)
                response = generate_answer(model, system_content)
                # print("---------------")
                # print(main_task)
                # print(subtask1)
                # print(subtask2)
                # print(response)
                judge = process_answer(response)
                if judge == False:
                    for i in range(5):
                        response = generate_answer(model, system_content)
                        judge = process_answer(response)
                        if judge != False:
                            break
                if judge != False:
                    all_edge_number += 1
                    if judge == "[yes]":
                        this_data_scores += 1
        this_data_scores = this_data_scores/all_edge_number
        data["scores"] = this_data_scores
        if this_data_scores >= gold_threshold:
            gold_data.append(data)
        elif (this_data_scores < gold_threshold) and (this_data_scores >= review_threshold):
            review_data.append(data)
        else:
            delete_data.append(data)
        print(f"--{main_task}-- : scores: {this_data_scores}")
    
    write_file(gold_data, gold_data_write_filepath)
    write_file(review_data, review_data_write_filepath)
    write_file(delete_data, delete_data_write_filepath)

    

        



