import json
from utils import *
from copy import deepcopy

def sort_subtasks(raw_subtasks):
    final_subtasks = []
    for i in range(len(raw_subtasks)):
        now_subtask_dict = raw_subtasks[i]
        temp_subtask_dict = {}
        temp_subtask_dict["stepId"] = i+1
        temp_subtask_dict["step"] = now_subtask_dict["step"]
        final_subtasks.append(temp_subtask_dict)
    return final_subtasks

if __name__ == '__main__':

    raw_filename = "/data/xzhang/task_planning/main/data/final_data/test.json"
    change_write_filename = "/data/xzhang/task_planning/main/data/final_data/test.json"

    raw_data = read_file(raw_filename)
    final_data = []
    for data in raw_data:
        now_data = deepcopy(data)
        now_data["substeps"] = sort_subtasks(data["substeps"])
        final_data.append(now_data)
    write_file(final_data, change_write_filename)

        

