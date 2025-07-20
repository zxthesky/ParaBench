from utils import *

import os

gold_data_main_tasks_foldername = "/data/xzhang/task_planning/main/data/new_main_task"
test_gold_main_tasks = read_data_from_folder(gold_data_main_tasks_foldername)


train_data_task_folder_name = "/data/xzhang/task_planning/main/data/train_main_task"

categories_name = os.listdir(train_data_task_folder_name)

all_main_tasks_number = 0


for category_name in categories_name:

    now_filneame = os.path.join(train_data_task_folder_name, category_name)

    final_now_train_main_tasks = []

    raw_now_train_main_tasks = read_file(now_filneame)

    for main_task in raw_now_train_main_tasks:
        if main_task not in test_gold_main_tasks and main_task not in final_now_train_main_tasks:
            final_now_train_main_tasks.append(main_task)

    write_file(final_now_train_main_tasks, now_filneame)

    all_main_tasks_number += len(final_now_train_main_tasks)

print(all_main_tasks_number)  



