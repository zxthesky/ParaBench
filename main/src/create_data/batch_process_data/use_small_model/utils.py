import json
import transformers
from copy import deepcopy

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

def write_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
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

def add_root_node(node):
    """
    给每一颗树加上一个 root node，防止出现对于一个任务，前几个子任务之间都是平衡的没有先后关系，这样一棵树就没有根节点，加入根节点，更方便之后的操作。
    """
    all_candidate_node = find_first_layer_node(node)
    if all_candidate_node == -1:
        return -1
    root_node = Node("")
    for candidate_node in all_candidate_node:
        root_node.children.append(candidate_node)
        candidate_node.parent.append(root_node)
    return root_node

def create_tree(all_edges, check_data=False):
    has_nodes = {}
    final_node = None
    for edge in all_edges:
        substep1 = edge["subtask1"]
        substep2 = edge["subtask2"]
        if has_nodes.get(substep1, -1) == -1:
            parent_node = Node(substep1)
            has_nodes[substep1] = parent_node
        else:
            parent_node = has_nodes[substep1]
        if has_nodes.get(substep2, -1) == -1:
            children_node = Node(substep2)
            has_nodes[substep2] = children_node
        else:
            children_node = has_nodes[substep2]
        
        parent_node.children.append(children_node)
        children_node.parent.append(parent_node)

        final_node = children_node
    # if check_data:
    #     print(final_node.information)
    root_node = add_root_node(final_node)
    return root_node

def get_tree_deepth(root_node):
    max_deepth = root_node.layer
    for node in root_node.children:
        node.layer = root_node.layer + 1
        max_deepth = max(max_deepth, get_tree_deepth(node))
    
    return max_deepth

def get_tree_width(root_node, layer_2_node: dict = {}, now_layer: int = 0):
    if layer_2_node.get(now_layer, -1) == -1:
        layer_2_node[now_layer] = []

    if root_node not in layer_2_node[now_layer]:
        layer_2_node[now_layer].append(root_node)

    for node in root_node.children:
        get_tree_width(node, layer_2_node, now_layer+1)


def get_tree_depth_and_width(root_node):
    root_node.layer = 0
    layer_id_num = {}
    depth = get_tree_deepth(root_node)
    get_tree_width(root_node, layer_id_num)
    # for i in layer_id_num:
    #     print("---------")
    #     print(i)
    #     for node in layer_id_num[i]:
    #         print(node.information)
    width = max([len(layer_id_num[i]) for i in layer_id_num])
    
    return depth, width