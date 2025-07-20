from tqdm import *
# from model.LLama3 import LLAMA3Model
import json
from utils import read_file, eval_all_data, write_file, create_tree_with_subtasks
from utils import *
import re
from copy import deepcopy
from prompt.prompt import *
from openai import OpenAI


all_wrong_model_type_number = 0  ######## 调用模型降级情况


class My_model():
    def __init__(self, model_name, model_type_name):
        self.model_name = model_name
        self.model_type_name = model_type_name

    def generate(self, messages, append_or_not=True):
        if self.model_name == "Qwen":
            now_model_type = self.model_type_name
            client = OpenAI(
            api_key="sk-4e49eda19925439e894bbb46dbe9ff6f",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            
            completion = client.chat.completions.create(
                model=now_model_type,
                messages=messages
            )
            
            if completion.model != self.model_type_name:
                all_wrong_model_type_number += 1
            raw_response = completion.choices[0].message.content
            return raw_response, completion.model
        elif self.model_name == "silicon":
            now_model_type = self.model_type_name
            client = OpenAI(
            api_key="sk-ybkpixhhmwdzcuicxgjfmmsjlttdcpspupubrcrnrojovjfr",
            base_url="https://api.siliconflow.cn/v1",
            )
            completion = client.chat.completions.create(
                model=now_model_type,
                messages=messages
            )
            raw_response = completion.choices[0].message.content
            return raw_response, completion.model
        elif self.model_name == "gptapi":

            now_model_type = self.model_type_name
            client = OpenAI(
            api_key="sk-ahkpJFOR5QMUHyBwDc1d8c141d3b4c57B2921e32Ff692fF6",
            base_url="https://www.gptapi.us/v1",
            )
            completion = client.chat.completions.create(
                model=now_model_type,
                messages=messages
            )
            raw_response = completion.choices[0].message.content
            return raw_response, completion.model


def change_Linear_2_dependencies(data: list)->dict:
    """将线性数据转化为边依赖
    Args:
        data (list): raw LLM's response
    Returns (dict): [{"subtask1":..., "subtask2":...}, ...]
    """
    dependenies = []
    for i in range(len(data)-1):
        temp_data = {}
        temp_data["subtask1"] = data[i]
        temp_data["subtask2"] = data[i+1]
        dependenies.append(temp_data)
    return dependenies

def dfs_get_layer(now_node: Node):

    for children_node in now_node.children:
        if children_node.layer != None:
            children_node.layer = max(now_node.layer + 1, children_node.layer)
        else:
            children_node.layer = now_node.layer + 1
        dfs_get_layer(children_node)

def layer_2_node(now_node: Node, has_node: dict):
    """
    Args:
        now_node(Node): now node
        has_node(dict): {1: [...], 2: [...]}
    """
    if now_node.layer not in has_node:
        has_node[now_node.layer] = [now_node.information]
    else:
        if now_node.information not in has_node[now_node.layer]:
            has_node[now_node.layer].append(now_node.information)
    
    for children_node in now_node.children:
        layer_2_node(children_node, has_node)

def change_dependencies_to_linear(gold_dependencies: list, subtasks: list):
    """将原来的依赖树转化为线性
    """
    has_node = {}  ##### 存储node信息，
    gold_answer_root_node = create_tree_with_subtasks(gold_dependencies, subtasks)
    if gold_answer_root_node == False:
        return False
    gold_answer_root_node.layer = 0
    dfs_get_layer(gold_answer_root_node)
    for now_node in gold_answer_root_node.children:
        layer_2_node(now_node, has_node)
    all_depth = len(has_node) 
    all_information = []
    for i in range(1, all_depth+1):
        all_information.extend(has_node[i])
    return all_information


def LLM_Linear(model, all_data: list, write_file_linear_filename: str)->dict:
    """需要模型线性回答问题

    Args:
        data (dict): [{"task": ..., "subtasks": [...]}, ...., "gold_dependencies": [...]]
    Returns(dict): final metric

    """
    try:
        has_generate_data = read_file(write_file_linear_filename)
        has_generate_data_de_dependencies = []
        for now_data in has_generate_data:
            has_generate_data_de_dependencies.append(now_data["gold_dependencies"])
    except:
        has_generate_data_de_dependencies = []

    un_process_data = []

    for tempdata in all_data:
        if tempdata["gold_dependencies"] not in has_generate_data_de_dependencies:
            un_process_data.append(tempdata)

    template_system_prompt = prompt_linear_1
    final_data = []
    init_wrong_number = 0
    for i in tqdm(range(len(un_process_data))):
        data = un_process_data[i]
        temp_data = {}
        main_task = data["task"]
        subtasks = data["subtasks"]
        system_prompt = deepcopy(template_system_prompt)
        user_info = deepcopy(user_linear_1)
        all_information = change_dependencies_to_linear(data["gold_dependencies"], subtasks)
        if all_information == False:
            continue
        print("***************************************")
        all_information = deepcopy(subtasks)
        # system_prompt = system_prompt.replace("(task)", main_task)
        # system_prompt = system_prompt.replace("(subtasks)", str(subtasks))
        # system_prompt = system_prompt.replace("(subtasks)", str(all_information))

        # all_information = all_information[::-1]  #### 子任务序列是否反过来


        user_info = user_info.replace("(task)", main_task)
        user_info = user_info.replace("(subtasks)", str(all_information))
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_info}]
        now_model_type = None
        try:
            raw_response, now_model_type = model.generate(messages)
            final_response = re.findall(r"\[.*\]", raw_response, re.DOTALL)[0]
            lst_data = json.loads(final_response)
        except:
            try:
                raw_response, now_model_type = model.generate(messages)
                final_response = re.findall(r"\[.*\]", raw_response, re.DOTALL)[0]
                lst_data = json.loads(final_response)
            except:
                init_wrong_number += 1
                continue
        dependencies = change_Linear_2_dependencies(lst_data)
        # dependencies = change_Linear_2_dependencies(all_information)
        temp_data["task"] = main_task
        temp_data["subtasks"] = subtasks
        temp_data["gold_dependencies"] = data["gold_dependencies"]
    
        temp_data["predict_dependencies"] = dependencies
        temp_data["now_model_type"] = now_model_type
        final_data.append(temp_data)
        write_file(final_data, write_file_linear_filename)

    metric, _, _ = eval_all_data(final_data, init_wrong_number)

    final_data.append({"metric": metric})
    write_file(final_data, write_file_linear_filename)
    return metric

##########################################
##########################################

def change_raw_dependencies_2_final_dependenies(main_task, raw_datas: list):
    """将原始模型输出处理成通用格式返回，将main_task对应的边去除了
    """
    dependenies = []
    for data in raw_datas:
        now_children_subtask = data["subtask"]
        if now_children_subtask != main_task:
            dependencies_information_lst = data["dependencies"]
            for parent_ifo in dependencies_information_lst:
                if parent_ifo != main_task:  #### 去掉这些main_task的边
                    temp_data = {}
                    temp_data["subtask1"] = parent_ifo
                    temp_data["subtask2"] = now_children_subtask
                    dependenies.append(temp_data)
    return dependenies


def LLM_tree(model, all_data: list, write_file_tree_filename: str)->dict:
    """让模型尽可能的并行返回

    Args:
        all_data (dict): [{"task": ..., "subtasks": [...]}, ...., "gold_dependencies": [...]]
    Returns(dict): final metric

    """
    try:
        has_generate_data = read_file(write_file_linear_filename)
        has_generate_data_de_dependencies = []
        for now_data in has_generate_data:
            has_generate_data_de_dependencies.append(now_data["gold_dependencies"])
    except:
        has_generate_data_de_dependencies = []

    un_process_data = []

    for tempdata in all_data:
        if tempdata["gold_dependencies"] not in has_generate_data_de_dependencies:
            un_process_data.append(tempdata)


    raw_system_prompt = prompt_tree_1
    final_data = []
    init_wrong_number = 0
    for i in tqdm(range(len(un_process_data))):
        data = un_process_data[i]
        temp_data = {}
        system_prompt = deepcopy(raw_system_prompt)
        user_info = deepcopy(user_tree_1)
        main_task = data["task"]
        subtasks = data["subtasks"]
        # system_prompt = system_prompt.replace("(maintask)", main_task)
        # system_prompt = system_prompt.replace("(all_sub_tasks)", str(subtasks))

        right_sequence_subtasks = change_dependencies_to_linear(data["gold_dependencies"], subtasks)  ### 获得正序的子任务序列

        # reversed_sequence_subtasks = right_sequence_subtasks[::-1]

        user_info = user_info.replace("(maintask)", main_task)


        # user_info = user_info.replace("(all_sub_tasks)", str(subtasks))
        user_info = user_info.replace("(all_sub_tasks)", str(right_sequence_subtasks))
        # user_info = user_info.replace("(all_sub_tasks)", str(reversed_sequence_subtasks))

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_info}]
        now_model_type = None
        try:
            raw_response, now_model_type = model.generate(messages)
            final_response = re.findall(r"\[.*\]", raw_response, re.DOTALL)[0]
            lst_data = json.loads(final_response)
        except:
            try:
                raw_response, now_model_type = model.generate(messages)
                final_response = re.findall(r"\[.*\]", raw_response, re.DOTALL)[0]
                lst_data = json.loads(final_response)
            except:
                init_wrong_number += 1
                continue
        try:
            dependencies = change_raw_dependencies_2_final_dependenies(main_task, lst_data)
            temp_data["task"] = main_task
            temp_data["subtasks"] = subtasks
            temp_data["gold_dependencies"] = data["gold_dependencies"]
            temp_data["predict_dependencies"] = dependencies
            temp_data["now_model_type"] = now_model_type
            final_data.append(temp_data)
            write_file(final_data, write_file_tree_filename)
        except:
            init_wrong_number += 1
    metric, _, _ = eval_all_data(final_data, init_wrong_number)

    final_data.append({"metric": metric})
    write_file(final_data, write_file_tree_filename)
    return metric

##########################################
##########################################

def check_raw_whether_valid(raw_data):
    substeps = []
    for substep in raw_data["substeps"]:
        if substep["step"] not in substeps:
            substeps.append(substep["step"])
        else:
            return False
        
    for edge in raw_data["dependencies"]:
        if (edge["subtask1"] == edge["subtask2"]) or (edge["subtask1"] not in substeps) or (edge["subtask2"] not in substeps):
            return False
    return True

def change_small_model_prediction_to_dependencies(raw_small_model_prediction):
    raw_prediction = raw_small_model_prediction["pred"] # 如[[0, 1], [1, 2], [1, 4], [2, 4], [3, 5], [4, 3], [4, 5], [5, 6], [5, 7], [6, 7]] 数字表示对应的subtask的id
    step_id_2_subtask = {}
    raw_all_subtasks = raw_small_model_prediction["raw"]["raw_data"]["substeps"]
    whether_valid = check_raw_whether_valid(raw_small_model_prediction["raw"]["raw_data"])
    if whether_valid == False:
        return False
    all_subtasks = []
    for subtask in raw_all_subtasks:
        step_id_2_subtask[int(subtask["stepId"])] = subtask["step"]
        all_subtasks.append(subtask["step"])
    dependencies = []
    for id_edge in raw_prediction:
        subtask1 = step_id_2_subtask[int(id_edge[0])+1]
        subtask2 = step_id_2_subtask[int(id_edge[1])+1]
        dependencies.append({"subtask1": subtask1, "subtask2": subtask2})

    return {
        "subtasks" : all_subtasks,
        "dependencies": dependencies
    }

def get_subtasks_from_subtasks_dict(subtask_dict):
    """将subtask_dict转化为subtasks[subtask1, subtask2...]
    """
    subtasks = []
    for subtask_dict_temp in subtask_dict:
        now_subtask = subtask_dict_temp["step"]
        subtasks.append(now_subtask)
    return subtasks


def use_small_model_eval(all_gold_data: list, small_model_predict_data, write_file_small_model_filename):
    """
    用小模型判别任务的顺序
    
    """
    final_data = []
    small_model_predict_data = small_model_predict_data
    main_task_2_gold_data = {}
    init_wrong_number = 0
    for data in all_gold_data:
        now_main_task = data["task"]
        subtasks = data["subtasks"]
        now_key = now_main_task + str(subtasks)
        main_task_2_gold_data[now_key] = data["gold_dependencies"]
    
    for data in small_model_predict_data:
        temp_data = {}
        raw_predict_data = data["raw"]["raw_data"]
        main_task = raw_predict_data["task"]
        now_subtask_dict = raw_predict_data["substeps"]
        subtasks = get_subtasks_from_subtasks_dict(now_subtask_dict)
 
        now_key = main_task + str(subtasks)

        gold_dependencies = main_task_2_gold_data[now_key]
        try:
            processed_small_model_prediction = change_small_model_prediction_to_dependencies(data)
            if processed_small_model_prediction == False:
                init_wrong_number += 1
                continue
        except:
            init_wrong_number += 1
            print("卧槽")
            continue
        subtasks, predict_dependencies = processed_small_model_prediction["subtasks"], processed_small_model_prediction["dependencies"]
        temp_data["task"] = main_task
        temp_data["subtasks"] = subtasks
        temp_data["gold_dependencies"] = gold_dependencies
        temp_data["predict_dependencies"] = predict_dependencies
        final_data.append(temp_data)

    metric, _, _ = eval_all_data(final_data, init_wrong_number)

    final_data.append({"metric": metric})
    write_file(final_data, write_file_small_model_filename)
    return metric

##########################################
##########################################

def change_raw_small_model_dependencies_to_prompt(main_task:str, raw_small_model_data_edge: list, subtasks: list):
    """
    raw_small_model_data: [{"subtask1": ..., "subtask2": ...}, ......]
    需要将其转化为         [{"subtask": ..., dependencies: [....]}]
    
    Args:
        main_task(str): main task
        raw_small_model_data(list) : raw_small_model_prediction
        subtasks : all_subtasks
    """

    get_all_subtask_dependenies = {}
    for subtask in subtasks:
        get_all_subtask_dependenies[subtask] = []

    for edge in raw_small_model_data_edge:
        assert edge["subtask2"] in subtasks
        get_all_subtask_dependenies[edge["subtask2"]].append(edge["subtask1"])
    for subtask in subtasks:
        if len(get_all_subtask_dependenies[subtask]) == 0:
            get_all_subtask_dependenies[subtask].append(main_task)
    final_data_we_need = []
    for subtask in get_all_subtask_dependenies:
        temp_data = {}
        temp_data["subtask"] = subtask
        temp_data["dependencies"] = get_all_subtask_dependenies[subtask]
        final_data_we_need.append(temp_data)
    return final_data_we_need

###############################################

def conbine_small_model_and_LLM(model, all_gold_data, small_model_predict_data, write_file_LLM_and_small_model_filename):
    """
    用小模型判别顺序来辅助大模型
    """

    try:
        has_generate_data = read_file(write_file_linear_filename)
        has_generate_data_de_dependencies = []
        for now_data in has_generate_data:
            has_generate_data_de_dependencies.append(now_data["gold_dependencies"])
    except:
        has_generate_data_de_dependencies = []

    un_process_data = []

    for tempdata in all_gold_data: 
        if tempdata["gold_dependencies"] not in has_generate_data_de_dependencies:
            un_process_data.append(tempdata)

    raw_system_prompt = prompt_combine_small_and_LLM_1

    small_model_predict_data = small_model_predict_data
    main_task_2_prediction_dependencies = {}
    init_wrong_number = 0
    for data in small_model_predict_data:
        processed_small_model_prediction = change_small_model_prediction_to_dependencies(data)
        main_task = data["raw"]["raw_data"]["task"]
        temp_subtasks = get_subtasks_from_subtask_dict(data["raw"]["raw_data"]["substeps"])
        now_key = main_task + str(temp_subtasks)
        if processed_small_model_prediction == False:
            main_task_2_prediction_dependencies[now_key] = None
        else:
            subtasks, predict_dependencies = processed_small_model_prediction["subtasks"], processed_small_model_prediction["dependencies"]
            main_task_2_prediction_dependencies[now_key] = predict_dependencies

    final_data = []
    for i in tqdm(range(len(un_process_data))):
        data = un_process_data[i]
        temp_data = {}
        main_task = data["task"]
        subtasks = data["subtasks"]

        right_sequence_subtasks = change_dependencies_to_linear(data["gold_dependencies"], subtasks)  ### 获得正序的子任务序列

        now_temp_key = main_task + str(subtasks)
        if main_task_2_prediction_dependencies[now_temp_key] == None:
            init_wrong_number += 1
            continue
        
        small_model_raw_data = main_task_2_prediction_dependencies[now_temp_key]
        small_model_dependencies = change_raw_small_model_dependencies_to_prompt(main_task, small_model_raw_data, subtasks)
        system_prompt = deepcopy(raw_system_prompt)
        user_info = deepcopy(user_combine_small_and_LLM_1)
        
        # system_prompt = system_prompt.replace("(maintask)", main_task)
        # system_prompt = system_prompt.replace("(all_sub_tasks)", str(subtasks))
        # system_prompt = system_prompt.replace("(raw_dependencies)", str(small_model_dependencies))
        user_info = user_info.replace("(maintask)", main_task)

        # user_info = user_info.replace("(all_sub_tasks)", str(subtasks))
        reversed_sequence_subtasks = right_sequence_subtasks[::-1]
        user_info = user_info.replace("(all_sub_tasks)", str(reversed_sequence_subtasks))

        user_info = user_info.replace("(raw_dependencies)", str(small_model_dependencies))
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_info}]
        now_model_type = None
        try:
            raw_response, now_model_type = model.generate(messages)
            final_response = re.findall(r"\[.*\]", raw_response, re.DOTALL)[0]
            lst_data = json.loads(final_response)
        except:
            try:
                raw_response, now_model_type = model.generate(messages)
                final_response = re.findall(r"\[.*\]", raw_response, re.DOTALL)[0]
                lst_data = json.loads(final_response)
            except:
                init_wrong_number += 1
                continue
        dependencies = change_raw_dependencies_2_final_dependenies(main_task, lst_data)
        temp_data["task"] = main_task
        temp_data["subtasks"] = subtasks
        temp_data["gold_dependencies"] = data["gold_dependencies"]
        # temp_data["predict_dependencies"] = small_model_raw_data
        temp_data["predict_dependencies"] = dependencies
        temp_data["now_model_type"] = now_model_type
        final_data.append(temp_data)
        write_file(final_data, write_file_LLM_and_small_model_filename)

    metric, _, _ = eval_all_data(final_data, init_wrong_number)

    final_data.append({"metric": metric})
    write_file(final_data, write_file_LLM_and_small_model_filename)
    return metric

def conbine_small_model_and_LLM_version_2(model, all_gold_data, small_model_predict_data, write_file_LLM_and_small_model_filename):
    """
    用小模型判别顺序来辅助大模型
    """

    try:
        has_generate_data = read_file(write_file_linear_filename)
        has_generate_data_de_dependencies = []
        for now_data in has_generate_data:
            has_generate_data_de_dependencies.append(now_data["gold_dependencies"])
    except:
        has_generate_data_de_dependencies = []

    un_process_data = []

    for tempdata in all_gold_data:
        if tempdata["gold_dependencies"] not in has_generate_data_de_dependencies:
            un_process_data.append(tempdata)

    raw_system_prompt = prompt_combine_small_and_LLM_version_2

    small_model_predict_data = small_model_predict_data
    main_task_2_prediction_dependencies = {}
    init_wrong_number = 0
    for data in small_model_predict_data:
        main_task = data["raw"]["raw_data"]["task"]
        temp_subtasks = get_subtasks_from_subtask_dict(data["raw"]["raw_data"]["substeps"])
        processed_small_model_prediction = change_small_model_prediction_to_dependencies(data)

        now_key = main_task + str(temp_subtasks)
        if processed_small_model_prediction == False:
            main_task_2_prediction_dependencies[now_key] = None
        else:
            subtasks, predict_dependencies = processed_small_model_prediction["subtasks"], processed_small_model_prediction["dependencies"]
            main_task_2_prediction_dependencies[now_key] = predict_dependencies
    
    final_data = []
    for i in tqdm(range(len(un_process_data))):
        data = un_process_data[i]
        temp_data = {}
        main_task = data["task"]
        subtasks = data["subtasks"]
        
        now_temp_key = main_task + str(subtasks)
        if main_task_2_prediction_dependencies[now_temp_key] == None:
            init_wrong_number += 1
            continue
        small_model_raw_data = main_task_2_prediction_dependencies[now_temp_key]
        small_model_dependencies = change_raw_small_model_dependencies_to_prompt(main_task, small_model_raw_data, subtasks)
        system_prompt = deepcopy(raw_system_prompt)
        user_info = deepcopy(user_combine_small_and_LLM_version_2)
        # system_prompt = system_prompt.replace("(maintask)", main_task)
        # system_prompt = system_prompt.replace("(all_sub_tasks)", str(subtasks))
        user_info = user_info.replace("(maintask)", main_task)
        user_info = user_info.replace("(all_sub_tasks)", str(subtasks))

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_info}]
        now_model_type = None
        try:
            raw_response, now_model_type = model.generate(messages)
            final_response = re.findall(r"\[.*\]", raw_response, re.DOTALL)[0]
            lst_data = json.loads(final_response)
        except:
            try:
                raw_response, now_model_type = model.generate(messages)
                final_response = re.findall(r"\[.*\]", raw_response, re.DOTALL)[0]
                lst_data = json.loads(final_response)
            except:
                init_wrong_number += 1
                continue
        Observation_prompt = Observation_prompt_version_2
        Observation_prompt = Observation_prompt.replace("(small_model_dependencies)", str(small_model_dependencies))
        messages.append({"role": "assistant", "content": raw_response})
        messages.append({"role": "user", "content": Observation_prompt})
        now_model_type = None
        try:
            raw_response, now_model_type = model.generate(messages)
            final_response = re.findall(r"\[.*\]", raw_response, re.DOTALL)[0]
            lst_data = json.loads(final_response)
        except:
            try:
                raw_response, now_model_type = model.generate(messages)
                final_response = re.findall(r"\[.*\]", raw_response, re.DOTALL)[0]
                lst_data = json.loads(final_response)
            except:
                init_wrong_number += 1
                continue
        dependencies = change_raw_dependencies_2_final_dependenies(main_task, lst_data)
        temp_data["task"] = main_task
        temp_data["subtasks"] = subtasks
        temp_data["gold_dependencies"] = data["gold_dependencies"]
        # temp_data["predict_dependencies"] = small_model_raw_data
        temp_data["predict_dependencies"] = dependencies
        temp_data["now_model_type"] = now_model_type
        final_data.append(temp_data)
        write_file(final_data, write_file_LLM_and_small_model_filename)

    metric, _, _ = eval_all_data(final_data, init_wrong_number)

    final_data.append({"metric": metric})
    write_file(final_data, write_file_LLM_and_small_model_filename)
    return metric


###########################
########################### 处理原始的gold data

def process_raw_golden_data(raw_all_data):
    final_data = []
    for data in raw_all_data:
        temp_data = {}
        temp_data["task"] = data["task"]
        subtasks = [i["step"] for i in data["substeps"]]
        temp_data["subtasks"] = subtasks
        temp_data["gold_dependencies"] = data["dependencies"]
        final_data.append(temp_data)
    return final_data

###########################
###########################



def main(model, raw_all_data, small_model_reference_data, write_file_linear_filename, write_file_tree_filename, write_file_small_model_filename, write_file_LLM_and_small_model_filename, mode=None):
    final_return = {}
    all_golden_data = process_raw_golden_data(raw_all_data)
    if mode["linear"] == True:
        linear_metric = LLM_Linear(model, all_golden_data, write_file_linear_filename)
        final_return["linear_metric"] = linear_metric
    if mode["tree"] == True:
        tree_metric = LLM_tree(model, all_golden_data, write_file_tree_filename)
        final_return["tree_metric"] = tree_metric
    if mode["small_model"] == True:
        small_model_metric = use_small_model_eval(all_golden_data, small_model_reference_data, write_file_small_model_filename)
        final_return["small_model_metric"] = small_model_metric
    if mode["small_combine_LLM"] == True:
        combine_small_and_LLM_metric = conbine_small_model_and_LLM(model, all_golden_data, small_model_reference_data, write_file_LLM_and_small_model_filename)
        final_return["conbine_small_and_LLM_metric"] = combine_small_and_LLM_metric
    
    return final_return

def get_small_model_predict_edge(filename):
    with open(filename, 'r') as f:
        all_data = json.loads(f)
    return all_data

def reduce_juhao(temp_data: str):
    if temp_data.endswith(".") or temp_data.endswith(",") or temp_data.endswith(" "):
        now_temp_data = temp_data[:-1]
        final_str_return = " "
        for i in range(len(now_temp_data)):
            if now_temp_data[i] != " ":
                final_str_return += now_temp_data[i]
            else:
                if final_str_return[-1] != " ":
                    final_str_return += now_temp_data[i]
        return final_str_return[1:]
    else:        
        now_temp_data = temp_data[:]
        final_str_return = " "
        for i in range(len(now_temp_data)):
            if now_temp_data[i] != " ":
                final_str_return += now_temp_data[i]
            else:
                if final_str_return[-1] != " ":
                    final_str_return += now_temp_data[i]
        return final_str_return[1:]
def process_raw_tasklama_data(all_raw_data):
    """去掉子任务中的句号等，统一格式
    """
    final_data = []
    for data in all_raw_data:
        temp_data = {}
        task = data["task"]
        temp_data["task"] = task
        temp_data["assumptions"] = data["assumptions"]
        subtasks = []
        for subtask in data["substeps"]:
            temp_id_subtask = {"stepId": subtask["stepId"], "step": reduce_juhao(subtask["step"])}
            subtasks.append(temp_id_subtask)
        temp_data["substeps"] = subtasks
        dependencies = []
        for edge in data["dependencies"]:
            if edge["subtask1"] != edge["subtask2"]:
                temp_edge = {"subtask1": reduce_juhao(edge["subtask1"]), "relation": edge["relation"], "subtask2": reduce_juhao(edge["subtask2"])}
                dependencies.append(temp_edge)
        temp_data["dependencies"] = dependencies
        final_data.append(temp_data)
    return final_data


if __name__ == "__main__":

    model_folder_name = "qwen2-72b-instruct"
    model_type_name = "qwen2-72b-instruct"
    mode = {"linear": False, "tree": False, "small_model": False, "small_combine_LLM": True}

    write_file_linear_filename = f"/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/{model_folder_name}/linear_random.json"
    write_file_tree_filename = f"/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/{model_folder_name}/tree_right_sequence.json"
    write_file_small_model_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/small_model.json"
    write_file_LLM_and_small_model_filename = f"/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/{model_folder_name}/LLM_conbine_small_model_reversed_sequence.json"

    # model_path = "/data/xzhang/model_parameter/LLama3-instruct"
    # model = LLAMA3Model(model_path)
    model = My_model("Qwen", model_type_name=model_type_name)
    raw_filename = "/public/home/hmqiu/xzhang/task_planning/main/data/final_data/test.json" ## 原始数据 ，task_planning的数据 所有的test集合
    raw_all_data = read_file(raw_filename)
    print(len(raw_all_data))

    raw_all_data = process_raw_tasklama_data(raw_all_data)
    small_model_reference_data_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/small_model_predict_data/test_predictions_3000_1005.json" ## 小模型推理后的结果
    small_model_reference_data = read_file(small_model_reference_data_filename)

    result = main(model, raw_all_data, small_model_reference_data, write_file_linear_filename, write_file_tree_filename, write_file_small_model_filename, write_file_LLM_and_small_model_filename, mode=mode)
    print(result)
    print("---------------")
    print(all_wrong_model_type_number)


