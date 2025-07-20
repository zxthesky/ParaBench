import openai
from openai import OpenAI
import json
from utils import create_tree_with_subtasks, get_tree_depth, get_tree_width, read_file
from utils import *
import re
import os
from tqdm import *

# openai.api_key = ""
# openai.api_base = "https://www.gptapi.us/v1"

def write_file(filename, data):
    with open(filename, 'w')as f:
        json.dump(data, f, indent=2)


def reduce_main_task_edge(dependencies:list, main_task:str):
    """去掉包含main_task的边，这些边在下面对比时候没有意义，只是为了后面画图方便"""
    dependencies_without_main_tasks = []
    for edge in dependencies:
        if edge["subtask1"].lower() == main_task.lower():
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
        if now_subtask not in subtasks:   
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
    root_node = create_tree_with_subtasks(dependencies_without_main_tasks, input_subtasks)
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
        if now_subtask not in subtasks:
            subtasks.append({"stepId": i+1, "step": now_subtask})
            if now_subtask == None:
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


def use_gpt_generate_data(client, main_tasks: list, has_generate_data: list, raw_answer_data: list, write_data_filename: str, raw_write_data_filename: str, raw_model_name:str="gpt-3.5-turbo"):

    system_content = """Complex task can often be divided into multiple subtasks to complete. These subtasks often have some dependency order, that is, this subtask must be completed after some subtasks are completed. At the same time, some tasks are not related to each other and can run in parallel at the same time. Now I will give you a task, and you need to divide it into multiple subtasks. The most important point is to improve the parallelism between subtasks. You need to make more subtasks depend on the same subtask, which means that many subtasks will appear in the "dependencies" of different subtasks. This means that when the current subtask is completed, it can choose one of multiple candidate non-dependency subtasks to complete in the next step. Your answer needs to be returned in JSON format: "[{"subtask": ... "dependencies": [....]}, ......]". "subtask" represents the current subtask and the number of subtasks should be between 10 and 20. "dependencies" is a list, indicating which subtasks the current subtask needs to be completed before it can be completed, that is, the current subtask depends on these tasks. You need to make many subtasks appear in the "dependencies" of different subtasks more than twice as much as possible, the more times the better.
    Here are some examples:

    Example 1:
    task: make dinner
    [{"subtask": "confirm menu", "dependencies": ["make dinner"]}, {"subtask": "wash rice", "dependencies": ["make dinner"]}, {"subtask": "set up the table", "dependencies": ["make dinner"]}, {"subtask": "prepare ingredients", "dependencies": [""confirm menu"]}, {"subtask": "cut vegetables", "dependencies": ["prepare ingredients"]}, {"subtask": "clean and slicing meat": , "dependencies": ["prepare ingredients"]}, {"subtask": "make soup", "dependencies": ["prepare ingredients"]}, {"subtask": "prepare dishes" , "dependencies": ["cut vegetables", "clean and slicing meat"]}, {"subtask": "cook rice", "dependencies": ["wash rice"]}, {"subtask": "place the plate", "dependencies": ["set up the table"]}, {"subtask": "place bowl and chopsticks", "dependencies": ["set up the table"]}, {"subtask": "serve the food", "dependencies": ["prepare dishes", "make soup", "cook rice", "place the plate", "place bowl and chopsticks"]}]

    Example 2:
    task: build a deck around your pool
    [{"subtask": "obtain building permits", "dependencies": ["build a deck around your pool"]}, {"subtask": "determine the shape and size of your deck", "dependencies": ["build a deck around your pool"]}, {"subtask": "select materials", "dependencies": ["build a deck around your pool"]}, {"subtask": "prepare relevant tools", "dependencies": ["build a deck around your pool"]}, {"subtask": "purchase the required materials", "dependencies": ["determine the shape and size of your deck", "select materials"]}, {"subtask": "clean the site", "dependencies": ["obtain building permits", "purchase the required materials", "prepare relevant tools"]}, {"subtask": "excavate the foundation pit", "dependencies": ["clean the site"]}, {"subtask": "build the deck frame", "dependencies": ["clean the site"]}, {"subtask": "install foundation supports(such as columns)", "dependenies": ["excavate the foundation pit"]}, {"subtask": "install horizontal support beams on the frame", "dependencies": ["build the deck frame"]}, {"subtask": "lay deck planks", "dependencies": ["install foundation supports(such as columns)", "install horizontal support beams on the frame"]}, {"subtask": "install guardrail", "dependencies": ["install foundation supports(such as columns)", "install horizontal support beams on the frame"]}, {"subtask": "clean and decorate deck", "dependencies": ["lay deck planks", "install guardrail"]}]

    task: """

    # system_content_version2 = """Complex task can often be divided into multiple subtasks to complete. These subtasks often have some dependency order, that is, this subtask must be completed after some subtasks are completed. At the same time, some tasks are not related to each other and can run in parallel at the same time. Now I will give you a task, and you need to divide it into multiple subtasks, so that the subtasks can run in parallel as much as possible(multiple subtasks depend on one subtask). The most important point is to improve the parallelism between subtasks.  Your answer needs to be returned in JSON format: "[{"subtask": ... "The next subtasks": [....]}, ......]". "subtask" represents the current subtask, and "The next subtasks" is a list that represents the candidate subtasks after the current subtask is completed. These candidate subtasks have no dependencies and can be completed in parallel. If "subtask" is null, the corresponding "The next subtasks" indicates the first batch of subtasks, which do not depend on other subtasks. If "The next subtasks" is null, it means that the current subtask has no subsequent subtasks that depend on it. You need to improve the parallelism between subtasks, that is, try to make the number of candidate tasks in "The next subtasks" as large as possible. Subtasks do not need to be too basic or complex.
    # Here are some examples:

    # Example 1:
    # task: make dinner
    # [{"subtask": null, "The next subtasks": ["confirm menu", "wash rice", "set up the table"]}, {"subtask": "confirm menu", "The next subtasks": ["prepare ingredients"]}, {"subtask": "prepare ingredients", "The next subtasks": ["cut vegetables", "clean and slicing meat", "make soup"]}, {"subtask": "cut vegetable", "The next subtasks": ["prepare dishes"]}, {"subtask": "clean and slicing meat", "The next subtasks": ["prepare dishes"]}, {"subtask": "wash rice", "The next subtasks": ["cook rice"]}, {"subtask": "set up the table", "The next subtasks": ["place the plate", "place bowl and chopsticks"]}, {"subtask": "prepare dishes", "The next subtasks": ["serve the food"]}, {"subtask": "make soup", "The next subtasks": ["serve the food"]}, {"subtask": "place the plate", "The next subtasks": ["serve the food"]}, {"subtask": "place bowl and chopsticks", "The next subtasks": ["serve the food"]}, {"subtask": "serve the food", "The next subtasks": null}]

    # Example 2:
    # task: build a deck around your pool
    # [{"subtask": null, "The next subtasks": ["obtain building permits", "determine the shape and size of your deck", "select materials", "prepare relevant tools"]}, {"subtask": "obtain building permits", "The next subtasks": ["clean the site"]}, {"subtask": "determine the shape and size of your deck", "The next subtasks": ["purchase the required materials"]}, {"subtask": "select materials", "The next subtasks": ["purchase the required materials"]}, {"subtask": "purchase the required materials", "The next subtasks": ["clean the site"]}, {"subtask": "obtain building permits", "The next subtasks": ["clean the site"]}, {"subtask": "prepare relevant tools", "The next subtasks": ["clean the site"]}, {"subtask": "clean the site", "The next subtasks": ["excavate the foundation pit", "build the deck frame"]}, {"subtask": "excavate the foundation pit", "The next subtasks": ["install foundation supports"]}, {"subtask": "install foundation supports", "The next subtasks": ["lay deck planks", "install guardrail"]}, {"subtask": "build the deck frame", "The next subtasks": ["install horizontal support beams on the frame"]}, {"subtask": "install horizontal support beams on the frame", "The next subtasks": ["lay deck planks", "install guardrail"]}, {"subtask": "lay deck planks", "The next subtasks": ["clean and decorate deck"]}, {"subtask": "install guardrail", "The next subtasks": ["clean and decorate deck"]}]

    # task: """



    raw_response_datas = []
    raw_response_datas.extend(raw_answer_data)
    all_data = []
    all_data.extend(has_generate_data)
    for i in tqdm(range(len(main_tasks))):
        print(f"----------------------  {i}  ----------------------------")
        {"role": "system", "content": system_content}
        main_task = main_tasks[i]
        system_content_final = system_content + main_task

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
            # try:
            #     completion = client.chat.completions.create(
            #     model=model_name,
            #     messages=messages
            #     )
            #     raw_response = completion.choices[0].message.content
            #     response = re.findall(r"\[.*\{.*\}.*\]", raw_response, re.DOTALL)[0]

            #     one_data = process_raw_response(response, main_task)
            # except:
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

            # revise_prompt_version2 = "I am not satisfied with your answer. The parallelism is not high (two subtasks can be performed at the same time without mutual dependence). You can increase parallelism by scaling certain subtasks. You need to re-divide the tasks and regenerate the answer. Remember to improve the parallelism of the subtasks, You need the subtask to have more candidate subtasks that depend on it, that is, the elements in \"The next subtasks\" need to be as many as possible. Finally you should return it in the Json format of the above example. Your answer needs to be returned in JSON format: \"[{\"subtask\": ... \"The next subtasks\": [....]}, ......]\""

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
                # try:
                #     completion = client.chat.completions.create(
                #     model=model_name,
                #     messages=messages
                #     )

                #     second_raw_response = completion.choices[0].message.content
                #     second_response = re.findall(r"\[.*\{.*\}.*\]", second_raw_response, re.DOTALL)[0]
                #     second_one_data = process_raw_response(second_response, main_task)
                # except:
                continue
            second_all_subtasks_number = len(second_one_data["substeps"])
            second_depth = second_one_data["depth"]
            raw_data["second_response"] = second_raw_response
            if (second_depth/second_all_subtasks_number) < (depth/all_subtasks_number):
                all_data.append(second_one_data)
            else:
                all_data.append(one_data)
        else:
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
    # write_data_filename = f"/data/xzhang/task_planning/create_data/data/{model_name}/all_data.json"
    # raw_write_data_filename = f"/data/xzhang/task_planning/create_data/data/{model_name}/raw_response.json"

    write_data_filename = f"/data/xzhang/task_planning/main/data/final_data/extra_8.json"
    raw_write_data_filename = f"/data/xzhang/task_planning/main/data/final_data/extra_8_data_raw.json"
    main_tasks_folder_name = "/data/xzhang/task_planning/main/data/8_data"

    main_tasks = get_main_tasks(main_tasks_folder_name)
    try:
        has_generate_main_tasks = get_has_generate_data("/data/xzhang/task_planning/main/data/final_data/extra_8.json")
    except:
        has_generate_main_tasks = []
    main_tasks = reduce_has_generate_data(main_tasks, has_generate_main_tasks)
    try:   
        has_generate_data = read_file(write_data_filename)
    except:
        has_generate_data = []
    try:
        raw_answer_data = read_file(raw_write_data_filename)
    except:
        raw_answer_data = []
    print(len(has_generate_data))
    print(len(main_tasks))
    print(len(raw_answer_data))
    # main_tasks = ["build a simple log house"]
    client = OpenAI(api_key="sk-C9Gzkape7lwmtKS2B2Ef4b2d3a7a4c3bB7A641AfB24397E4", base_url="https://www.gptapi.us/v1")
    use_gpt_generate_data(client, main_tasks, has_generate_data, raw_answer_data, write_data_filename, raw_write_data_filename, model_name)
