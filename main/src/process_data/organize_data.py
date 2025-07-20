"""
看数据的时候会对数据进行修改，所以这个代码是用来重新整理那些杂乱的数据
"""
from utils import read_file, write_file

def detect_subtasks(raw_subtasks, dependencies, main_task, index):
    """
    检查是否subtasks中的子任务是否和dependencies一致

    Return:
        1:表示正常
        0:表示子任务和边对应的任务不一样
        -1:表示边出现了和一开始定义的子任务不一样的边

    """
    judge = 0
    # if main_task == 'Conduct a commercial energy audit':
    #     judge = 1
    #     print(index)
    subtasks = []
    for subtask_dict in raw_subtasks:
        subtasks.append(subtask_dict["step"])

    dependencies_subtasks = []
    for edge in dependencies:
        subtask1 = edge["subtask1"]
        subtask2 = edge["subtask2"]
        if subtask1.lower() != main_task.lower():
            if subtask1 in subtasks:
                if subtask1 not in dependencies_subtasks:
                    dependencies_subtasks.append(subtask1)
            else:
                if judge != 1:
                    print(subtask1)
                return -1
        if subtask2.lower() != main_task.lower():
            if subtask2 in subtasks:
                if subtask2 not in dependencies_subtasks:
                    dependencies_subtasks.append(subtask2)
            else:
                if judge != 1:
                    print(subtask2)
                return -1
            
    if len(dependencies_subtasks) != len(subtasks):
        print("??????????")
        print(len(dependencies_subtasks))
        print(len(subtasks))
        for j in subtasks:
            if j not in dependencies_subtasks:
                print(j)
        return 0
    
    return 1

def sort_subtask_id(raw_subtasks):
    final_subtasks = []
    for i in range(len(raw_subtasks)):
        temp_subtask_dict = {}
        temp_subtask_dict["stepId"] = i+1
        temp_subtask_dict["step"] = raw_subtasks[i]["step"]
        final_subtasks.append(temp_subtask_dict)
    return final_subtasks



def detect_translated_index(main_task, translated_datas):
    for i in range(len(translated_datas)):
        if translated_datas[i]["task"] == main_task:
            return i

def main(raw_data_filename, output_data_filename):
    wrong_data_main_task = []
    raw_datas = read_file(raw_data_filename)
    # raw_translated_datas = read_file(translated_filename)
    raw_translated_main_tasks = []

    # print(len(raw_translated_datas))
    # for translated_data in raw_translated_datas:
    #     raw_translated_main_tasks.append(translated_data["task"])
    
    # preserved_datas = []
    # for data in raw_datas:
    #     if data["task"] in raw_translated_main_tasks:
    #         preserved_datas.append(data)

    final_data = []

    # print(len(preserved_datas))

    for i in range(len(raw_datas)):
        data = raw_datas[i]
        temp_data = {}
        main_task = data["task"]
        temp_data["task"] = main_task
        temp_data["assumptions"] = data["assumptions"]
        now_subtasks = data["substeps"]
        now_dependencies = data["dependencies"]

        detect_number = detect_subtasks(now_subtasks, now_dependencies, main_task, i)

        if detect_number != 1:
            if detect_number == 0:
                print(detect_number)
                # print(f"picture index: {detect_translated_index(main_task, raw_translated_datas)}")
                print(main_task)
                print("===================     end      =========================")
            print("???????????")
            wrong_data_main_task.append(main_task)


        
        
        temp_data["substeps"] = sort_subtask_id(now_subtasks)
        temp_data["dependencies"] = now_dependencies

        final_data.append(temp_data)

    print(len(final_data))

    # write_file(final_data, output_data_filename)
    # print(len(wrong_data_main_task))
    # print(wrong_data_main_task)




        


if __name__ == "__main__":
    raw_data_filename = "/data/xzhang/task_planning/main/data/final_data/extra_8.json"
    
    output_data_filename = "/data/xzhang/task_planning/main/data/final_data/adsa.json"

    # translated_filename = "/data/xzhang/task_planning/main/process_data/processed_data/final_data_852.json"

    main(raw_data_filename, output_data_filename)

