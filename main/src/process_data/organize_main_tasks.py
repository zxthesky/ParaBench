import os
from utils import *


def detect_get_all_tasks(filename_2_dict, all_main_tasks):
    """
    用来检查主任务文件是否包含了所有的主要任务
    """
    all_compare_main_tasks = []
    for i in filename_2_dict:
        all_compare_main_tasks.extend(filename_2_dict[i])

    un_appear_main_tasks = []
    for task in all_main_tasks:
        if task not in all_compare_main_tasks:
            un_appear_main_tasks.append(task)
    print(len(all_compare_main_tasks))

    test_data = []
    for task in all_main_tasks:
        if task not in test_data:
            test_data.append(task)
        else:
            print(task)

    return un_appear_main_tasks

if __name__ == "__main__":
    old_main_task_folder = "/data/xzhang/task_planning/main/data/main_tasks_old"
    new_main_task_folder = "/data/xzhang/task_planning/main/data/new_main_task"

    preserve_data_filename = "/data/xzhang/task_planning/main/data/final_data/test_826.json"
    preserve_datas = read_file(preserve_data_filename)
    all_preserved_main_tasks = []
    
    for data in preserve_datas:
        all_preserved_main_tasks.append(data["task"])

    print(len(all_preserved_main_tasks))

    all_old_filenames = os.listdir(old_main_task_folder)

    filename_2_main_task = {}

    for now_filename in all_old_filenames:
        if filename_2_main_task.get(now_filename, -1) == -1:
            filename_2_main_task[now_filename] = []
        
        old_file_path = os.path.join(old_main_task_folder, now_filename)
        old_main_tasks= read_file(old_file_path)
        for task in old_main_tasks:
            if task in all_preserved_main_tasks:
                filename_2_main_task[now_filename].append(task)

        new_write_file_path = os.path.join(new_main_task_folder, now_filename)
        write_file(filename_2_main_task[now_filename], new_write_file_path)
    
    unappear_main_tasks = detect_get_all_tasks(filename_2_main_task, all_preserved_main_tasks)

    # print(unappear_main_tasks)






