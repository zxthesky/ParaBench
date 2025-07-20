import json
import transformers
from copy import deepcopy
import os
from eval import *
import re


def read_file(filename):
    if filename.endswith(".json"):
        with open(filename, 'r') as f:
            all_data = json.load(f)
        return all_data
    elif filename.endswith(".jsonl"):
        all_data = []
        with open(filename, 'r') as f:
            for line in f:
                data = json.loads(line)
                all_data.append(data)
        return all_data
    else:
        print("your filename is wrong")
def read_data_from_folder(folder_name):
    all_filenames = os.listdir(folder_name)

    all_data = []
    for filename in all_filenames:
        now_file_path = os.path.join(folder_name, filename)
        now_data = read_file(now_file_path)
        all_data.extend(now_data)
    return all_data

def write_file(all_data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

class Node:
    def __init__(self, information: str = None) -> None:
        self.information: str = information
        self.children: list = []
        self.parent: list = []
        self.layer: int = None

def find_first_layer_node(node: Node)->list:

    if node.parent == []:
        return [node]
    else:
        final_return_lst = []
        for parent_node in node.parent:
            lst = find_first_layer_node(parent_node)
            for temp_node in lst:
                if temp_node not in final_return_lst:
                    final_return_lst.append(temp_node)
        return final_return_lst

# def add_root_node(node):
#     """
#     给每一颗树加上一个 root node，防止出现对于一个任务，前几个子任务之间都是平衡的没有先后关系，这样一棵树就没有根节点，加入根节点，更方便之后的操作。
#     """
#     all_candidate_node = find_first_layer_node(node)
#     if all_candidate_node == -1:
#         return -1
#     root_node = Node("")
#     for candidate_node in all_candidate_node:
#         root_node.children.append(candidate_node)
#         candidate_node.parent.append(root_node)
#     return root_node

# def create_tree(all_edges, check_data=False):
#     has_nodes = {}
#     final_node = None
#     for edge in all_edges:
#         substep1 = edge["subtask1"]
#         substep2 = edge["subtask2"]
#         if has_nodes.get(substep1, -1) == -1:
#             parent_node = Node(substep1)
#             has_nodes[substep1] = parent_node
#         else:
#             parent_node = has_nodes[substep1]
#         if has_nodes.get(substep2, -1) == -1:
#             children_node = Node(substep2)
#             has_nodes[substep2] = children_node
#         else:
#             children_node = has_nodes[substep2]
        
#         parent_node.children.append(children_node)
#         children_node.parent.append(parent_node)

#         final_node = children_node
#     # if check_data:
#     #     print(final_node.information)
#     root_node = add_root_node(final_node)
#     return root_node


def add_root_node(has_nodes: dict):
    """
    Args:
        has_node(dict): {information: node, ............}
    """
    all_candidate_node = []
    for information in has_nodes:
        now_node = has_nodes[information]
        if len(now_node.parent) == 0:
            all_candidate_node.append(now_node)
    if len(all_candidate_node) == 0:
        return -1
    root_node = Node()
    for candidate_node in all_candidate_node:
        root_node.children.append(candidate_node)
        candidate_node.parent.append(root_node)
    return root_node


def get_tree_depth(root_node):
    max_deepth = root_node.layer
    for node in root_node.children:
        node.layer = root_node.layer + 1
        max_deepth = max(max_deepth, get_tree_depth(node))
    
    return max_deepth
def get_tree_width(root_node: Node):
    """将原来的依赖树转化为线性
    """
    has_node = {}  ##### 存储node信息，
    gold_answer_root_node = root_node
    if gold_answer_root_node == False:
        return False
    gold_answer_root_node.layer = 0
    dfs_get_layer(gold_answer_root_node)
    for now_node in gold_answer_root_node.children:
        layer_2_node(now_node, has_node)

    max_width = 0
    for i in has_node:
        max_width = max(max_width, len(has_node[i]))

    return max_width

# def get_tree_width(root_node, layer_2_node: dict = {}, now_layer: int = 0):
#     if layer_2_node.get(now_layer, -1) == -1:
#         layer_2_node[now_layer] = []

#     if root_node not in layer_2_node[now_layer]:
#         layer_2_node[now_layer].append(root_node)

#     for node in root_node.children:
#         get_tree_width(node, layer_2_node, now_layer+1)


# def get_tree_depth_and_width(root_node):
#     root_node.layer = 0
#     layer_id_num = {}
#     depth = get_tree_deepth(root_node)
#     get_tree_width(root_node, layer_id_num)
#     width = max([len(layer_id_num[i]) for i in layer_id_num])
    
#     return depth, width

def get_subtasks_from_dict(subtasks_dict):
    subtasks = []
    for subtask_dict in subtasks_dict:
        subtasks.append(subtask_dict["step"])
    return subtasks

def convert_raw_dependencies_to_icl_dependencies(subtasks, raw_dependencies):
    
    final_icl_dependencies = []
    icl_dependecies_dict = {}
    for dependency in raw_dependencies:
        subtask1 = dependency["subtask1"]
        subtask2 = dependency["subtask2"]
        if subtask2 not in icl_dependecies_dict:
            icl_dependecies_dict[subtask2] = [subtask1]
        else:
            icl_dependecies_dict[subtask2].append(subtask1)
    
    for subtask in subtasks:
        if subtask in icl_dependecies_dict:
            final_icl_dependencies.append({"subtask": subtask, "dependencies": icl_dependecies_dict[subtask]})
    
    return final_icl_dependencies


def process_raw_generate_data_dependencies(raw_dependencies, main_task):
    """原始生成中dependence加入了main_task的依赖边，会对评价有一些影响，现在需要将其去掉
    """
    final_data = []
    for subtask_dict in raw_dependencies:
        temp_subtask_dict = {}
        now_subtask = subtask_dict["subtask"]
        temp_subtask_dict["subtask"] = now_subtask
        now_dependencies = subtask_dict["dependencies"]
        dependencies_without_main_task = []
        for now_dependency in now_dependencies:
            if now_dependency.lower() != main_task.lower():
                dependencies_without_main_task.append(now_dependency)
        temp_subtask_dict["dependencies"] = dependencies_without_main_task
        final_data.append(temp_subtask_dict)
    return final_data



def eval_data_quality(main_task, raw_response, processed_data):

    response = re.findall(r"\[.*\{.*\}.*\]", raw_response, re.DOTALL)[0]   
    raw_subtasks = [subtask_dict["step"] for subtask_dict in processed_data["substeps"]]
    final_subtasks = change_dependencies_to_linear(processed_data["dependencies"], raw_subtasks, processed_data["task"])
    formed_data = json.loads(response)

    pass_or_not = get_response(main_task, final_subtasks)

    formed_data_without_main_task = process_raw_generate_data_dependencies(formed_data, processed_data["task"])
    # print("------------------")
    # print(processed_data["task"])
    # print("------------------")
    # print(formed_data)
    # print("------------------")
    # print(formed_data_without_main_task)
    score = get_response(main_task, formed_data_without_main_task, eval_model="scores")

    return pass_or_not, score






################################
################################
#####                                   下面的是将线性数据转化为边的格式，或者将一个树的数据转化为线性                       ###########



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


def reduce_main_task_edge(dependencies:list, main_task:str):
    """去掉包含main_task的边，这些边在下面对比时候没有意义，只是为了后面画图方便"""
    dependencies_without_main_tasks = []
    for edge in dependencies:
        if (edge["subtask1"].lower()) == (main_task.lower()):
            continue
        else:
            dependencies_without_main_tasks.append(edge)
    return dependencies_without_main_tasks

def change_dependencies_to_linear(raw_gold_dependencies: list, subtasks: list, main_task):
    """将原来的依赖树转化为线性
    """
    has_node = {}  ##### 存储node信息，
    gold_dependencies = reduce_main_task_edge(raw_gold_dependencies, main_task)
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

################################
################################

def create_tree_with_subtasks(all_edges, subtasks, check_data=False):
    """构成树
    Args:
        all_edges: [{"subtask1":..., "subtask2": ...}, ....]
        subtasks: ["subtask1", ..."subtaski", ...]
    """

    has_nodes = {} ######3 information: node, 字典，information对应其node
    
    for information in subtasks:
        has_nodes[information] = Node(information)
    
    for edge in all_edges:
        substep1 = edge["subtask1"]
        substep2 = edge["subtask2"]
    
        assert has_nodes.get(substep1, -1) != -1
        assert has_nodes.get(substep2, -1) != -1
        parent_node = has_nodes[substep1]
        children_node = has_nodes[substep2]
        
        parent_node.children.append(children_node)
        children_node.parent.append(parent_node)
    # if check_data:
    #     print(final_node.information)
    root_node = add_root_node(has_nodes)

    whether_a_tree = check_whether_contain_circle(has_nodes) # True表示有环，不是一个树
    if whether_a_tree:
        return False
    else:
        return root_node
    

def check_one_data_whether_circle(node: Node)->bool:
    """判断环是否是当前节点产生
    Args:
        node(Node): now_node
    Returns(bool): True means produce a circle
    """
    for children in node.children:
        if judge_parent_node(node, children):
            return True
    return False

def check_whether_contain_circle(has_node: dict)->bool:
    """
    Args:
        has_node(dict): {"information": node, ......} 所有的节点，需要对每个节点去判别
    Returns(bool): True means has circle
    """
    for information in has_node:
        now_node = has_node[information]
        if check_one_data_whether_circle(now_node):
            return True
        
    return False
def judge_parent_node(now_node: Node, parent_node: Node):
    """判断parent_node是否在now_node节点之上
    """
    if len(now_node.parent) == 0:
        return False
    elif now_node == parent_node:
        return True
    else:
        for parent in now_node.parent:
            judge = judge_parent_node(parent, parent_node)
            if judge:
                return True
        return False