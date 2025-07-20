"""
生成的数据可能有些问题，需要格式化一下
比如去掉不需要的边和将子任务都最小化
"""
from utils import *

filename = "/public/home/hmqiu/xzhang/task_planning/main/data/train_data_3000/train_data_3000_all_edge.json"

raw_datas = read_file(filename)

final_datas = []

def lower_subtasks(subtasks_dict):
    final_data = []
    for data in subtasks_dict:
        temp_data = {}
        temp_data["stepId"] = data["stepId"]
        temp_data["step"] = data["step"].lower()
        final_data.append(temp_data)
    return final_data

def change_dependencies(main_task, raw_dependencies):
    """
    去掉主任务的边和将子任务小写
    """
    final_edges = []
    for edge in raw_dependencies:
        subtask1 = edge["subtask1"].lower()
        subtask2 = edge["subtask2"].lower()
        temp_edge = {}
        if subtask1 != main_task.lower() and subtask2 != main_task.lower():
            temp_edge["subtask1"] = subtask1
            temp_edge["relation"] = "Must be done before"
            temp_edge["subtask2"] = subtask2
            final_edges.append(temp_edge)
    return final_edges

for data in raw_datas:
    temp_data = {}
    main_task = data["task"]
    temp_data["task"] = data["task"]
    temp_data["assumptions"] = data["assumptions"]
    temp_data["substeps"] = lower_subtasks(data["substeps"])
    temp_data["dependencies"] = change_dependencies(main_task, data["dependencies"])
    temp_data["depth"] = data["depth"]
    temp_data["width"] = data["width"]

    final_datas.append(temp_data)

write_filename = "/public/home/hmqiu/xzhang/task_planning/main/data/train_data_3000/train_data_3000.json"

print(len(final_datas))
write_file(final_datas, write_filename)



