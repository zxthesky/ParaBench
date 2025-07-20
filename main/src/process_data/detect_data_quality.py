from utils import *

'''查看生成的边中的子任务是否在所有子任务中'''

if __name__ == '__main__':
    filename = "/data/xzhang/task_planning/main/data/final_data/extra_8.json"

    all_data = read_file(filename)
    print(len(all_data))
    for i in range(len(all_data)):
        data = all_data[i]
        main_task = data["task"]
        subtasks = data["substeps"]
        edges = data["dependencies"]

        edge_subtasks = []

        for edge in edges:
            subtask1 = edge["subtask1"]
            subtask2 = edge["subtask2"]

            if subtask1 not in edge_subtasks:
                edge_subtasks.append(subtask1)
            
            if subtask2 not in edge_subtasks:
                edge_subtasks.append(subtask2)

        if (len(edge_subtasks)-1) != (len(subtasks)):
            print("-------------------------")
            print(main_task)
            print(i)
            print(len(edge_subtasks))
            print(len(subtasks))


