import openai
from openai import OpenAI
import json
from utils import *
import re
import os
from tqdm import *
import random
from eval import *

gold_test_tasks_folder_name = "/public/home/hmqiu/xzhang/task_planning/main/data/new_main_task"


# openai.api_key = ""
# openai.api_base = "https://www.gptapi.us/v1"

def write_file(filename, data):
    with open(filename, 'w')as f:
        json.dump(data, f, indent=2)

def reduce_main_task_edge(dependencies:list, main_task:str):
    """去掉包含main_task的边，这些边在下面对比时候没有意义，只是为了后面画图方便"""
    dependencies_without_main_tasks = []
    for edge in dependencies:
        if (edge["subtask1"].lower()) == (main_task.lower()):
            continue
        else:
            dependencies_without_main_tasks.append(edge)
    return dependencies_without_main_tasks


def process_raw_response(response: str, main_task: str):
    temp_data = {}
    temp_data["task"] = main_task
    formed_data = json.loads(response)
    temp_data["assumptions"] = []
    subtasks = []
    dependencies = []
    for i in range(len(formed_data)):
        subtask_dict = formed_data[i]
        try:
            now_subtask = subtask_dict["subtask"].lower()
        except:
            now_subtask = None
        if subtask_dict["subtask"] not in subtasks:
            subtasks.append({"stepId": i+1, "step": now_subtask})
            if (subtask_dict["dependencies"] != None) and (subtask_dict["dependencies"] != []):
                for former_task in subtask_dict["dependencies"]:
                    dependencies.append({"subtask1": former_task.lower(), "relation": "Must be done before", "subtask2": now_subtask})
            else:
                dependencies.append({"subtask1": main_task, "relation": "Must be done before", "subtask2": now_subtask})
    temp_data["substeps"] = subtasks
    temp_data["dependencies"] = dependencies
    input_subtasks = [subtask_dict["step"] for subtask_dict in subtasks]
    dependencies_without_main_tasks = reduce_main_task_edge(dependencies, main_task)
    try:
        root_node = create_tree_with_subtasks(dependencies_without_main_tasks, input_subtasks)
    except:
        print(main_task)
        print(subtasks)
        print(formed_data)
    root_node.layer = 0
    depth = get_tree_depth(root_node)
    width = get_tree_width(root_node)
    temp_data["depth"] = depth
    temp_data["width"] = width
    return temp_data 

def process_raw_response_second_version(response: str, main_task: str):
    temp_data = {}
    temp_data["task"] = main_task
    formed_data = json.loads(response)
    temp_data["assumptions"] = []
    subtasks = []
    dependencies = []
    for i in range(len(formed_data)):
        subtask_dict = formed_data[i]
        try:
            now_subtask = subtask_dict["subtask"].lower()
        except:
            now_subtask = None
        if subtask_dict["subtask"] not in subtasks:
            subtasks.append({"stepId": i+1, "step": now_subtask})
            if subtask_dict["subtask"] == None:
                for later_task in subtask_dict["The next subtasks"]:
                    dependencies.append({"subtask1": main_task, "relation": "Must be done before", "subtask2": later_task.lower()})
            elif subtask_dict["The next subtasks"] == None:
                continue
            else:
                for later_task in subtask_dict["The next subtasks"]:
                    assert subtask_dict["subtask"] != None
                    dependencies.append({"subtask1": now_subtask, "relation": "Must be done before", "subtask2": later_task.lower()})
    temp_data["substeps"] = subtasks
    temp_data["dependencies"] = dependencies
    input_subtasks = [subtask_dict["step"] for subtask_dict in subtasks]
    dependencies_without_main_tasks = reduce_main_task_edge(dependencies, main_task)
    root_node = create_tree_with_subtasks(dependencies_without_main_tasks, input_subtasks)
    root_node.layer = 0
    depth = get_tree_depth(root_node)
    width = get_tree_width(root_node)
    temp_data["depth"] = depth
    temp_data["width"] = width
    return temp_data 

def category_2_task(foldername):
    """
    task属于哪一个类别
    Returns:
        dict : {"cooking": [task1, task2...] ... }
    """
    categories = os.listdir(foldername)
    category_2_task = {}
    for category_json in categories:
        category = category_json[:-5]
        now_filename = os.path.join(foldername, category_json)
        now_tasks = read_file(now_filename)
        category_2_task[category] = now_tasks
    
    return category_2_task


def get_candidate_icl_example(now_task, gold_category_2_tasks, need_generate_category_2_tasks):
    all_categories = list(gold_category_2_tasks.keys())
    
    icl_task = []
    for category in need_generate_category_2_tasks:
        if now_task in need_generate_category_2_tasks[category]:
            all_category_tasks = gold_category_2_tasks[category]
            random.shuffle(all_category_tasks)
            icl_task.append(all_category_tasks[0])

            random.shuffle(all_categories)
            if all_categories[0] != category:
                unrelevant_category = gold_category_2_tasks[all_categories[0]]
                random.shuffle(unrelevant_category)
                icl_task.append(unrelevant_category[0])
            return icl_task
            




def use_gpt_generate_data(client, main_task_foldername:str, main_tasks: list, has_generate_data: list, raw_answer_data: list, write_data_filename: str, raw_write_data_filename: str, candidate_raw_data_filename:str, raw_model_name:str="gpt-3.5-turbo"):
    """
    Args:
        client: client
        main_task(list): tasks we need to generate subtasks for
        has_generate_data(list): has generated datas. To prevent generate duplicate datas
        raw_answer_data(list): has generated datas. raw model outputs
        write_data_filename(str): filename we write generated datas
        raw_write_data_filename(str): write model raw responses
        candidate_raw_data_filename(str): correcct datas. we need them as icl examples
        aw_model_name(str): model you used
    """
    raw_candidate_data = read_file(candidate_raw_data_filename)

    gold_category_2_tasks = category_2_task(gold_test_tasks_folder_name)
    need_generate_category_2_tasks = category_2_task(main_task_foldername)


    system_content = """The complex task can often be divided into multiple subtasks to complete. These subtasks often have some dependency order, that is, this subtask must be completed after some subtasks are completed. At the same time, some tasks are not related to each other and can run in parallel at the same time. Now I will give you a task, and you need to divide it into multiple subtasks. The most important point is to improve the parallelism between subtasks. You need to make more subtasks depend on the same subtask, which means that many subtasks will appear in the "dependencies" of different subtasks. This means that when the current subtask is completed, it can choose one of multiple candidate non-dependency subtasks to complete in the next step. Your answer needs to be returned in JSON format: "[{"subtask": ... "dependencies": [....]}, ......]". "subtask" represents the current subtask and the number of subtasks should be between 10 and 20. "dependencies" is a list, indicating which subtasks the current subtask needs to be completed before it can be completed, that is, the current subtask depends on these tasks. You need to make many subtasks appear in the "dependencies" of different subtasks more than twice as much as possible, the more times the better.
    Here are some examples:

    Example 1:
    task: [query_1]
    [dependencies_1]

    Example 2:
    task: [query_2]
    [dependencies_2]

    task: """

    raw_response_datas = []
    raw_response_datas.extend(raw_answer_data)
    all_data = []
    all_data.extend(has_generate_data)
    for i in tqdm(range(len(main_tasks))):
        {"role": "system", "content": system_content}
        main_task = main_tasks[i]
        system_content_final = system_content + main_task
        
        icl_tasks = get_candidate_icl_example(main_task, gold_category_2_tasks, need_generate_category_2_tasks)
        icl_data = []
        
        for data in raw_candidate_data:
            if data["task"] in icl_tasks:
                icl_data.append(data)
                if len(icl_data) == 2:
                    break
        if icl_data[0]["task"] != icl_tasks[0] :
            temp_data = icl_data[0]
            icl_data[0] = icl_data[1]
            icl_data[1] = temp_data
       
        for j in range(len(icl_data)):
            now_data = icl_data[j]
            raw_dependencies = now_data["dependencies"]
            subtasks = get_subtasks_from_dict(now_data["substeps"])
            final_dependencies = convert_raw_dependencies_to_icl_dependencies(subtasks, raw_dependencies)
            now_query = f"[query_{j+1}]"
            now_dependecies = f"[dependencies_{j+1}]"
            system_content_final = system_content_final.replace(now_query, now_data["task"])
            system_content_final = system_content_final.replace(now_dependecies, str(final_dependencies))

        messages = [{"role": "system", "content": system_content_final}]
        
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            raw_response = completion.choices[0].message.content
            response = re.findall(r"\[.*\{.*\}.*\]", raw_response, re.DOTALL)[0]

            one_data = process_raw_response(response, main_task)
        except:
            continue

        raw_data = {}
        raw_data["task"] = main_task
        raw_data["first_response"] = raw_response

        all_subtasks_number = len(one_data["substeps"])
        depth = one_data["depth"]
        rate = 0.5
        if depth/all_subtasks_number > 0.7:
            
            # print(f"--------------   {i}    ---------------")

            messages.append({"role": "assistant", "content": raw_response})
            revise_prompt = "I am not satisfied with your answer. The parallelism is not high (two subtasks can be performed at the same time without mutual dependence). You need to re-divide the tasks and regenerate the answer. Remember to improve the parallelism of the subtasks, you need to make more subtasks depend on the same subtask, which means that more subtasks will appear in the \"dependencies\" of different subtasks. Finally you should return it in the Json format of the above example. Your answer needs to be returned in JSON format: \"[{\"subtask\": ... \"dependencies\": [....]}, ......]\""

            messages.append({"role": "user", "content": revise_prompt})
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages
                )

                second_raw_response = completion.choices[0].message.content
                second_response = re.findall(r"\[.*\{.*\}.*\]", second_raw_response, re.DOTALL)[0]
                second_one_data = process_raw_response(second_response, main_task)
            except:
                continue
            second_all_subtasks_number = len(second_one_data["substeps"])
            second_depth = second_one_data["depth"]
            raw_data["second_response"] = second_raw_response
            if (second_depth/second_all_subtasks_number) < (depth/all_subtasks_number):
                pass_or_not, score = eval_data_quality(main_task, second_raw_response, second_one_data)
                second_one_data["pass_or_not"] = pass_or_not
                second_one_data["score"] = score
                all_data.append(second_one_data)
            else:
                pass_or_not, score = eval_data_quality(main_task, raw_response, one_data)
                one_data["pass_or_not"] = pass_or_not
                one_data["score"] = score
                all_data.append(one_data)
        else:
            pass_or_not, score = eval_data_quality(main_task, raw_response, one_data)
            one_data["pass_or_not"] = pass_or_not
            one_data["score"] = score
            all_data.append(one_data)
            
        raw_response_datas.append(raw_data)
    
        write_file(write_data_filename, all_data)
        write_file(raw_write_data_filename, raw_response_datas)
    write_file(write_data_filename, all_data)
    write_file(raw_write_data_filename, raw_response_datas)

def get_main_tasks(folder_name):
    filenames  = os.listdir(folder_name)
    all_main_tasks = []
    for filename in filenames:
        filepath = os.path.join(folder_name, filename)
        now_main_tasks = read_file(filepath)
        all_main_tasks.extend(now_main_tasks)
    return all_main_tasks

def get_has_generate_data(filename):
    all_data = read_file(filename)
    has_generate_main_tasks = []
    for data in all_data:
        has_generate_main_tasks.append(data["task"])
    return has_generate_main_tasks

def reduce_has_generate_data(raw_data, has_generate_data):
    need_generate_data = []
    for data in raw_data:
        if data not in has_generate_data:
            need_generate_data.append(data)
    return need_generate_data

if __name__ == "__main__":
    model_name = "gpt-4o-2024-08-06"
    # model_name = "gpt-4-turbo"
    write_data_filename = f"/public/home/hmqiu/xzhang/task_planning/main/data/train_data_3000/train_data_3000.json"
    raw_write_data_filename = f"/public/home/hmqiu/xzhang/task_planning/main/data/train_data_3000/raw_response.json"
    main_tasks_folder_name = "/public/home/hmqiu/xzhang/task_planning/main/data/train_main_task_3000"
    candidate_raw_data_filename = "/public/home/hmqiu/xzhang/task_planning/main/data/final_data/data_contain_all_edge.json"

    main_tasks = get_main_tasks(main_tasks_folder_name)
    try:
        has_generate_main_tasks = get_has_generate_data(write_data_filename)
    except:
        has_generate_main_tasks = []
    main_tasks = reduce_has_generate_data(main_tasks, has_generate_main_tasks)

    try:
        has_generate_data = read_file(write_data_filename)
        raw_answer_data = read_file(raw_write_data_filename)
    except:
        has_generate_data = []
        raw_answer_data = []
    print(len(has_generate_data))
    print(len(main_tasks))
    print(len(raw_answer_data))
    # random.shuffle(main_tasks)
    # main_tasks = ["build a simple log house"]'
    client = OpenAI(api_key="sk-C9Gzkape7lwmtKS2B2Ef4b2d3a7a4c3bB7A641AfB24397E4", base_url="https://www.gptapi.us/v1")
    use_gpt_generate_data(client, main_tasks_folder_name, main_tasks, has_generate_data, raw_answer_data, write_data_filename, raw_write_data_filename, candidate_raw_data_filename, model_name)
