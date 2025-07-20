"""
将人工检测的test用来进行简单的测试
"""


from utils import *
import os
import random


if __name__ == "__main__":
    folder_name = "/data/xzhang/task_planning/main/data/new_main_task"
    all_filenames = os.listdir(folder_name)

    train_data_filename = "/data/xzhang/task_planning/main/data/test_data/train.json"
    dev_data_filename = "/data/xzhang/task_planning/main/data/test_data/dev.json"
    test_data_filename = "/data/xzhang/task_planning/main/data/test_data/test.json"

    all_datas = read_file("/data/xzhang/task_planning/main/data/final_data/test.json")
    print(len(all_datas))

    train_tasks = []
    dev_tasks = []
    test_tasks = []

    for filename in all_filenames:
        now_filename = os.path.join(folder_name, filename)
        now_tasks = read_file(now_filename)
        random.shuffle(now_tasks)
        all_number = len(now_tasks)
        if all_number < 10:
            test_tasks.append(now_tasks[0])
            dev_tasks.append(now_tasks[1])
            train_tasks.extend(now_tasks[2:])
        else:
            test_number = all_number//10
            dev_task_number = test_number
            test_tasks.extend(now_tasks[:test_number])
            dev_tasks.extend(now_tasks[test_number: 2*test_number])
            train_tasks.extend(now_tasks[2*test_number:])

    train_datas = []
    dev_datas = []
    test_datas = []

    for data in all_datas:
        main_task = data["task"]
        if main_task in test_tasks:
            test_datas.append(data)
        else:
            if main_task in dev_tasks:
                dev_datas.append(data)
            else:
                train_datas.append(data)
    print(len(train_datas))
    print(len(dev_datas))
    print(len(test_datas))

    write_file(train_datas, train_data_filename)
    write_file(dev_datas, dev_data_filename)
    write_file(test_datas, test_data_filename)


