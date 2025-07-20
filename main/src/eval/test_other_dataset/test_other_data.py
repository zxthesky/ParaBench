from datasets import load_dataset, load_from_disk
from utils import create_tree, get_tree_deepth, get_tree_deepth_and_width
import re
import json
test_dataset = load_from_disk("/data/xzhang/task_planning/main/eval/test_other_dataset/data/worbench_test")["test"]
train_dataset = load_from_disk("/data/xzhang/task_planning/main/eval/test_other_dataset/data/worbench_train")["train"]
train_data_dependencies = []

def contains_digit(s):
    return any(char.isdigit() for char in s)

source_to_data = {}

all_depth = 0
all_data_number = 0
all_subtasks_number = 0

for now_data in test_dataset:
    now_source = now_data["source"]
    if now_source not in source_to_data:
        source_to_data[now_source] = {"depth": 0, "number": 0, "all_subtasks_number": 0}
    now_dependencies = []
    all_subtasks = []
    final_edge = []
    conversations = now_data['conversations']
    final_pharse_we_need = conversations[-1]["content"]
    pattern = r"\([^()]*,[^()]*\)"
    edges = re.findall(pattern, final_pharse_we_need)
    for i in range(len(edges)):
        if "START" in edges[i]:
            edges = edges[i:]
            break
    for i in range(len(edges)):
        if contains_digit(edges[i]) == False:
            edges = edges[:i]
            break
    for edge_str in edges:
        edge_str = edge_str.replace(" ", "")
        if edge_str != "":
            parent, children = edge_str[1:-1].split(",")
            temp_edge = {}
            temp_edge["subtask1"] = parent
            temp_edge["subtask2"] = children
            if parent not in all_subtasks:
                all_subtasks.append(parent)
            if children not in all_subtasks:
                all_subtasks.append(children)
            now_dependencies.append(temp_edge)

    tree_root = create_tree(now_dependencies, all_subtasks)
    if tree_root != False:
        all_data_number += 1
        depth, width, information_2_layer = get_tree_deepth_and_width(tree_root)
        all_depth += depth
        all_subtasks_number += len(all_subtasks)
        source_to_data[now_source]["number"] = source_to_data[now_source]["number"] + 1
        source_to_data[now_source]["depth"] = source_to_data[now_source]["depth"] + depth
        source_to_data[now_source]["all_subtasks_number"] = source_to_data[now_source]["all_subtasks_number"] + len(all_subtasks)

print("---------------------")
print("----------all average depth-------------")
print((all_depth/all_data_number))
print((all_subtasks_number/all_data_number))
print((all_depth/all_data_number)/(all_subtasks_number/all_data_number))
print(">>>>>>>>>>>>>>>>>>>>>>")
for dataset in source_to_data:
    print(f"**********{dataset}************")
    print((source_to_data[dataset]["depth"]/source_to_data[dataset]["number"]))
    print(source_to_data[dataset]["all_subtasks_number"]/source_to_data[dataset]["number"])
    print((source_to_data[dataset]["depth"]/source_to_data[dataset]["number"])/(source_to_data[dataset]["all_subtasks_number"]/source_to_data[dataset]["number"]))

