'''分析下金标树中深度，广度的情况，设计分析性实验

max_width = 11
min_width = 2

max_width_rate = 0.733
min_width_rate = 0.2

max_depth = 12
min_depth = 2

max_depth_rate = 0.8
min_depth_rate = 0.181


'''
from utils import *

raw_gold_test_data_filename = "/public/home/hmqiu/xzhang/task_planning/main/data/final_data/test.json"

raw_gold_data = read_file(raw_gold_test_data_filename)

width_areas = {"level1": 0, "level2": 0, "level3": 0, "level4":0}

depth_areas = {"level1": 0, "level2": 0, "level3": 0, "level4":0}

task_width_rate = []

for data in raw_gold_data:
    subtasks = get_subtasks_from_subtask_dict(data["substeps"])
    gold_dependecies = data["dependencies"]


    gold_answer_root_node = create_tree_with_subtasks(gold_dependecies, subtasks)
    gold_depth, gold_width, gold_information_2_layer = get_tree_deepth_and_width(gold_answer_root_node)

    now_width_rate = gold_width/len(subtasks)
    now_depth_rate = gold_depth/len(subtasks)
    # if now_width_rate<0.35:
    #     width_areas["level1"] += 1
    # elif now_width_rate>=0.35 and now_width_rate<0.45:
    #     width_areas["level2"] += 1
    # elif now_width_rate>=0.45 and now_width_rate<0.5 :
    #     width_areas["level3"] += 1
    # else:
    #     width_areas["level4"] += 1
    if now_depth_rate<0.5:
        depth_areas["level1"] += 1
    elif now_depth_rate>=0.5 and now_depth_rate<0.54:
        depth_areas["level2"] += 1
    elif now_depth_rate>=0.54 and now_depth_rate<0.61 :
        depth_areas["level3"] += 1
    else:
        depth_areas["level4"] += 1


print(depth_areas)










    
