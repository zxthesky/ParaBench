import json


from utils import *

def get_width_with_raw_dependencies_and_subtasks(gold_dependencies: list, subtasks: list):
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

    max_width = 0
    for i in has_node:
        max_width = max(max_width, len(has_node[i]))

    return max_width

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


def get_width(raw_data):
    dependencies = raw_data["dependencies"]
    subtasks = [subtask_dict["step"] for subtask_dict in raw_data["substeps"]]
    max_width = get_width_with_raw_dependencies_and_subtasks(dependencies, subtasks)
    return max_width

if __name__ == '__main__':
    filename = "/data/xzhang/task_planning/main/data/final_data/test_wrong_width.json"
    write_filename = "/data/xzhang/task_planning/main/data/final_data/test.json"
    raw_datas = read_file(filename)
    final_data = []
    for data in raw_datas:
        new_width = get_width(raw_data=data)
        data["width"] = new_width
        final_data.append(data)

    write_file(final_data, write_filename)




