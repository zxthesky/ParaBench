from utils import *

if __name__ == "__main__":
    all_tasks = read_data_from_folder("/public/home/hmqiu/xzhang/task_planning/main/data/train_main_task")

    all_un_task = []

    for task in all_tasks:
        if task not in all_un_task:
            all_un_task.append(task)
    print(len(all_un_task))
    print(len(all_tasks))


