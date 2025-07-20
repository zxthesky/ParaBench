from utils import *

from copy import deepcopy

def process_dependencies(raw_dependencies, main_task):

    final_dependencies = []
    for edge in raw_dependencies:
        subtask1 = edge["subtask1"]
        if subtask1.lower() != main_task.lower():
            final_dependencies.append(edge)

    return final_dependencies


if __name__ == "__main__":
    filename_826 = "/data/xzhang/task_planning/main/data/medidate_data/test_826.json"
    filename_171 = "/data/xzhang/task_planning/main/data/medidate_data/cook_data_171.json"
    filename_8 = "/data/xzhang/task_planning/main/data/medidate_data/extra_8.json"

    write_file_name = "/data/xzhang/task_planning/main/data/final_data/test.json"

    data_826 = read_file(filename_826)
    data_171 = read_file(filename_171)
    data_8 = read_file(filename_8)

    all_data = data_826 + data_171 + data_8

    final_data = []

    for data in all_data:
        temp_data = deepcopy(data)
        main_task = data["task"]
        root_node = create_tree(temp_data["dependencies"])
        depth, width = get_tree_depth_and_width(root_node)
        temp_data["depth"] = depth -1 
        temp_data["width"] = width
        temp_data["dependencies"] = process_dependencies(data["dependencies"], main_task)
        final_data.append(temp_data)

    write_file(final_data, write_file_name)

    print(len(all_data))

