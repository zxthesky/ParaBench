from utils import *
import json
import os



####################### 这部分是tree的数据

Qwen_25_0_5b_tree_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2.5-0.5b-instruct/tree.json"
Qwen_25_1_5b_tree_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2.5-1.5b-instruct/tree.json"
Qwen_25_3b_tree_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2.5-3b-instruct/tree.json"
Qwen_25_7b_tree_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2.5-7b-instruct/tree.json"
Qwen_25_14b_tree_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2.5-14b-instruct/tree.json"
Qwen_25_32b_tree_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2.5-32b-instruct/tree.json"
Qwen_25_72b_tree_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2.5-72b-instruct/tree.json"

Qwen_2_72b_tree_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2-72b-instruct/tree.json"
Qwen_2_57b_tree_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2-57b-a14b-instruct/tree.json"
Qwen_2_7b_tree_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2-7b-instruct/tree.json"

llama_3_3_70b_tree_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/Llama-3.3-70B-Instruct/tree.json"

gpt_4o_tree_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/gpt-4o-2024-11-20/tree.json"
gpt_4o_mini_tree_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/gpt-4o-mini-2024-07-18/tree.json"
gpt_3_5_tree_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/gpt-3.5-turbo-0125/tree.json"

deepseek_2_5_tree_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/DeepSeek-V2.5/tree.json"
deepseek_3_tree_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/DeepSeek-V3/tree.json"

Qwen_25_0_5b_tree = read_file(Qwen_25_0_5b_tree_filename)
Qwen_25_1_5b_tree = read_file(Qwen_25_1_5b_tree_filename)
Qwen_25_3b_tree = read_file(Qwen_25_3b_tree_filename)
Qwen_25_7b_tree = read_file(Qwen_25_7b_tree_filename)
Qwen_25_14b_tree = read_file(Qwen_25_14b_tree_filename)
Qwen_25_32b_tree = read_file(Qwen_25_32b_tree_filename)
Qwen_25_72b_tree = read_file(Qwen_25_72b_tree_filename)
Qwen_2_72b_tree = read_file(Qwen_2_72b_tree_filename)
Qwen_2_57b_tree = read_file(Qwen_2_57b_tree_filename)
Qwen_2_7b_tree = read_file(Qwen_2_7b_tree_filename)
llama_3_3_70b_tree = read_file(llama_3_3_70b_tree_filename)
gpt_4o_tree = read_file(gpt_4o_tree_filename)
gpt_4o_mini_tree = read_file(gpt_4o_mini_tree_filename)
gpt_3_5_tree = read_file(gpt_3_5_tree_filename)
deepseek_2_5_tree = read_file(deepseek_2_5_tree_filename)
deepseek_3_tree = read_file(deepseek_3_tree_filename)


###################### 下面是 LLM_combine_small_model的数据

Qwen_25_0_5b_LLM_conbined_small_model_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2.5-0.5b-instruct/LLM_conbine_small_model.json"
Qwen_25_1_5b_LLM_conbined_small_model_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2.5-1.5b-instruct/LLM_conbine_small_model.json"
Qwen_25_3b_LLM_conbined_small_model_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2.5-3b-instruct/LLM_conbine_small_model.json"
Qwen_25_7b_LLM_conbined_small_model_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2.5-7b-instruct/LLM_conbine_small_model.json"
Qwen_25_14b_LLM_conbined_small_model_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2.5-14b-instruct/LLM_conbine_small_model.json"
Qwen_25_32b_LLM_conbined_small_model_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2.5-32b-instruct/LLM_conbine_small_model.json"
Qwen_25_72b_LLM_conbined_small_model_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2.5-72b-instruct/LLM_conbine_small_model.json"

Qwen_2_72b_LLM_conbined_small_model_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2-72b-instruct/LLM_conbine_small_model.json"
Qwen_2_57b_LLM_conbined_small_model_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2-57b-a14b-instruct/LLM_conbine_small_model.json"
Qwen_2_7b_LLM_conbined_small_model_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2-7b-instruct/LLM_conbine_small_model.json"

llama_3_3_70b_LLM_conbined_small_model_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/Llama-3.3-70B-Instruct/LLM_conbine_small_model.json"

gpt_4o_LLM_conbined_small_model_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/gpt-4o-2024-11-20/LLM_conbine_small_model.json"
gpt_4o_mini_LLM_conbined_small_model_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/gpt-4o-mini-2024-07-18/LLM_conbine_small_model.json"
gpt_3_5_LLM_conbined_small_model_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/gpt-3.5-turbo-0125/LLM_conbine_small_model.json"

deepseek_2_5_LLM_conbined_small_model_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/DeepSeek-V2.5/LLM_conbine_small_model.json"
deepseek_3_LLM_conbined_small_model_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/DeepSeek-V3/LLM_conbine_small_model.json"

Qwen_25_0_5b_LLM_conbined_small_model = read_file(Qwen_25_0_5b_LLM_conbined_small_model_filename)
Qwen_25_1_5b_LLM_conbined_small_model = read_file(Qwen_25_1_5b_LLM_conbined_small_model_filename)
Qwen_25_3b_LLM_conbined_small_model = read_file(Qwen_25_3b_LLM_conbined_small_model_filename)
Qwen_25_7b_LLM_conbined_small_model = read_file(Qwen_25_7b_LLM_conbined_small_model_filename)
Qwen_25_14b_LLM_conbined_small_model = read_file(Qwen_25_14b_LLM_conbined_small_model_filename)
Qwen_25_32b_LLM_conbined_small_model = read_file(Qwen_25_32b_LLM_conbined_small_model_filename)
Qwen_25_72b_LLM_conbined_small_model = read_file(Qwen_25_72b_LLM_conbined_small_model_filename)
Qwen_2_72b_LLM_conbined_small_model = read_file(Qwen_2_72b_LLM_conbined_small_model_filename)
Qwen_2_57b_LLM_conbined_small_model = read_file(Qwen_2_57b_LLM_conbined_small_model_filename)
Qwen_2_7b_LLM_conbined_small_model = read_file(Qwen_2_7b_LLM_conbined_small_model_filename)
llama_3_3_70b_LLM_conbined_small_model = read_file(llama_3_3_70b_LLM_conbined_small_model_filename)
gpt_4o_LLM_conbined_small_model = read_file(gpt_4o_LLM_conbined_small_model_filename)
gpt_4o_mini_LLM_conbined_small_model = read_file(gpt_4o_mini_LLM_conbined_small_model_filename)
gpt_3_5_LLM_conbined_small_model = read_file(gpt_3_5_LLM_conbined_small_model_filename)
deepseek_2_5_LLM_conbined_small_model = read_file(deepseek_2_5_LLM_conbined_small_model_filename)
deepseek_3_LLM_conbined_small_model = read_file(deepseek_3_LLM_conbined_small_model_filename)



####################################################
####################################################
####################################################
####################################################
'''对自己生成的tree的分析性实验'''
'''对自己生成的LLM_combined_small_model的分析性实验'''

################ 先获得域对应的task 如

category_to_tasks_folder_name = "/public/home/hmqiu/xzhang/task_planning/main/data/new_main_task"

gold_test_filename = "/public/home/hmqiu/xzhang/task_planning/main/data/final_data/test.json"

gold_test_data = read_file(gold_test_filename)
print(len(gold_test_data))
category_names = os.listdir(category_to_tasks_folder_name)
print(category_names)

category_name_to_task = {}   ########### 用来表示某个类里卖弄有多少个task

task_to_category_name = {}   ####### 给个task是属于哪一个类别，有些task属于多喝类别 {"task": ["bussiness", "activities"]}

all_categories = []

for category_name_json in category_names:
    now_category_name = category_name_json[:-5]   ####### 当前的类别名称

    all_categories.append(now_category_name)
    now_filename = os.path.join(category_to_tasks_folder_name, category_name_json)

    now_category_tasks = read_file(now_filename)

    category_name_to_task[now_category_name] = now_category_tasks
    for task in now_category_tasks:
        if task_to_category_name.get(task, -1) == -1:
            task_to_category_name[task] = [now_category_name]
        else:
            task_to_category_name[task].append(now_category_name)






################


def get_analysis_result(datas_needed_eval, task_to_category_name, all_categories):

    category_pass_and_all_datas = {}

    deepth_pass_and_all_datas = {"[0, 0.5)": [0, 0], "[0.5, 0.54)": [0, 0], "[0.54, 0.61)": [0, 0], "(0.61, 1]":[0, 0]}
    width_pass_and_all_datas = {"[2, 4)": [0, 0], "[4, 6)": [0, 0], "[6, 8)": [0, 0], "[8, 11]":[0, 0]}
    

    for category in all_categories:
        category_pass_and_all_datas[category] = [0, 0]

    for data in datas_needed_eval:
        task = data["task"]
        task_belong_categorys = task_to_category_name[task]   #### 得到当前task对应的类别 是list类型

        for category in task_belong_categorys:
            category_pass_and_all_datas[category][1] += 1

        gold_answer_root_node = create_tree_with_subtasks(data["gold_dependencies"], data["subtasks"], main_task=data["task"])
        gold_depth, gold_width, gold_information_2_layer = get_tree_deepth_and_width(gold_answer_root_node)
        now_width_rate = gold_width/len(data["subtasks"])
        now_depth_rate = gold_depth/len(data["subtasks"])


        # if deepth_pass_and_all_datas.get(gold_depth, -1) == -1:
        #     deepth_pass_and_all_datas[gold_depth] = [0, 1]
        # else:
        #     deepth_pass_and_all_datas[gold_depth][1] += 1
        if now_depth_rate<0.5:
            deepth_pass_and_all_datas["[0, 0.5)"][1] += 1
        elif now_depth_rate>=0.5 and now_depth_rate<0.54:
            deepth_pass_and_all_datas["[0.5, 0.54)"][1] += 1
        elif now_depth_rate>=0.54 and now_depth_rate<0.61 :
            deepth_pass_and_all_datas["[0.54, 0.61)"][1] += 1
        else:
            deepth_pass_and_all_datas["(0.61, 1]"][1] += 1

        
        # if width_pass_and_all_datas.get(gold_width, -1) == -1:
        #     width_pass_and_all_datas[gold_width] = [0, 1]
        # else:
        #     width_pass_and_all_datas[gold_width][1] += 1

        if gold_width<4:
            width_pass_and_all_datas["[2, 4)"][1] += 1
        elif gold_width>=4 and gold_width<6:
            width_pass_and_all_datas["[4, 6)"][1] += 1
        elif gold_width>=6 and gold_width<8 :
            width_pass_and_all_datas["[6, 8)"][1] += 1
        else:
            width_pass_and_all_datas["[8, 11]"][1] += 1

        whether_generate_wrong_subtask_judge = whether_generate_wrong_subtask(data["subtasks"], data["predict_dependencies"])
        if whether_generate_wrong_subtask_judge == False:
            
            predict_answer_root_node = create_tree_with_subtasks(data["predict_dependencies"], data["subtasks"])
            if gold_answer_root_node == False:
                continue
            if predict_answer_root_node != False: #### 确保生成了树
                
                raw_metric = compute_metric_one_data(gold_answer_root_node, predict_answer_root_node, data["subtasks"])

                if raw_metric["pass_or_not"] == True:
                    for category_temp in task_belong_categorys:
                        category_pass_and_all_datas[category_temp][0] += 1

                    # deepth_pass_and_all_datas[gold_depth][0] += 1
                    if now_depth_rate<0.5:
                        deepth_pass_and_all_datas["[0, 0.5)"][0] += 1
                    elif now_depth_rate>=0.5 and now_depth_rate<0.54:
                        deepth_pass_and_all_datas["[0.5, 0.54)"][0] += 1
                    elif now_depth_rate>=0.54 and now_depth_rate<0.61 :
                        deepth_pass_and_all_datas["[0.54, 0.61)"][0] += 1
                    else:
                        deepth_pass_and_all_datas["(0.61, 1]"][0] += 1
                                


                    if gold_width<4:
                        width_pass_and_all_datas["[2, 4)"][0] += 1
                    elif gold_width>=4 and gold_width<6:
                        width_pass_and_all_datas["[4, 6)"][0] += 1
                    elif gold_width>=6 and gold_width<8 :
                        width_pass_and_all_datas["[6, 8)"][0] += 1
                    else:
                        width_pass_and_all_datas["[8, 11]"][0] += 1

    return category_pass_and_all_datas, deepth_pass_and_all_datas, width_pass_and_all_datas       


def caculate_rate(raw_data):
    '''raw_data 是字典 {1: []}'''
    final_data = {}
    for data_key in raw_data:
        if raw_data[data_key][1] != 0:
            final_data[data_key] = raw_data[data_key][0]/raw_data[data_key][1]
        else:
            final_data[data_key] = 0
    return final_data

def main(tree_datas, LLM_combined_small_datas, task_to_category_name, all_categories):
    if "metric" in list(tree_datas[-1].keys()):
        tree_datas = tree_datas[:-1]

    if "metric" in list(LLM_combined_small_datas[-1].keys()):
        LLM_combined_small_datas = LLM_combined_small_datas[:-1]

    tree_category_pass_and_all_datas = {}   #### 对应模型原始的tree生成，存储每个类别对应的情况，如{"writing": [right_number, wrong_number]}
    tree_deepth_pass_and_all_datas = {} ### 存储每个deepth对应的情况，如{"writing": [right_number, wrong_number]}
    tree_width_pass_and_all_datas = {} ### 存储每个deepth对应的情况，如{"writing": [right_number, wrong_number]}

    tree_category_pass_and_all_datas, tree_deepth_pass_and_all_datas, tree_width_pass_and_all_datas = get_analysis_result(tree_datas, task_to_category_name, all_categories)

    LLM_combined_small_model_category_pass_and_all_datas = {}   #### 对应LLM结合小模型的生成，存储每个类别对应的情况，如{"writing": [right_number, wrong_number]}
    LLM_combined_small_model_deepth_pass_and_all_datas = {}  #### 存储每个deepth对应的情况，如{"writing": [right_number, wrong_number]}
    LLM_combined_small_model_width_pass_and_all_datas = {}

    LLM_combined_small_model_category_pass_and_all_datas, LLM_combined_small_model_deepth_pass_and_all_datas, LLM_combined_small_model_width_pass_and_all_datas = get_analysis_result(LLM_combined_small_datas, task_to_category_name, all_categories)


    tree_category_pass_rate = caculate_rate(tree_category_pass_and_all_datas)
    tree_deepth_pass_rate = caculate_rate(tree_deepth_pass_and_all_datas)
    tree_width_pass_rate = caculate_rate(tree_width_pass_and_all_datas)

    LLM_combined_small_model_category_pass_rate = caculate_rate(LLM_combined_small_model_category_pass_and_all_datas)
    LLM_combined_small_model_deepth_pass_rate = caculate_rate(LLM_combined_small_model_deepth_pass_and_all_datas)
    LLM_combined_small_model_width_pass_rate = caculate_rate(LLM_combined_small_model_width_pass_and_all_datas)



    return tree_category_pass_rate

    return {"tree": [tree_category_pass_rate, tree_deepth_pass_rate, tree_width_pass_rate], "LLM_combined_small_model": [LLM_combined_small_model_category_pass_rate, LLM_combined_small_model_deepth_pass_rate, LLM_combined_small_model_width_pass_rate]}





####################################################
####################################################
####################################################
####################################################

deepseek_3_metric = main(deepseek_3_tree, deepseek_3_LLM_conbined_small_model, task_to_category_name, all_categories)
gpt_4o_metric = main(gpt_4o_tree, gpt_4o_LLM_conbined_small_model, task_to_category_name, all_categories)
llama_3_3_70B_metric = main(llama_3_3_70b_tree, llama_3_3_70b_LLM_conbined_small_model, task_to_category_name, all_categories)
qwen_25_72B_metric = main(Qwen_25_72b_tree, Qwen_25_72b_LLM_conbined_small_model, task_to_category_name, all_categories)

Qwen_25_0_5b_metric = main(Qwen_25_0_5b_tree, Qwen_25_0_5b_LLM_conbined_small_model, task_to_category_name, all_categories)
Qwen_25_1_5b_metric = main(Qwen_25_1_5b_tree, Qwen_25_1_5b_LLM_conbined_small_model, task_to_category_name, all_categories)
Qwen_25_3b_metric = main(Qwen_25_3b_tree, Qwen_25_3b_LLM_conbined_small_model, task_to_category_name, all_categories)
Qwen_25_7b_metric = main(Qwen_25_7b_tree, Qwen_25_7b_LLM_conbined_small_model, task_to_category_name, all_categories)
Qwen_25_14b_metric = main(Qwen_25_14b_tree, Qwen_25_14b_LLM_conbined_small_model, task_to_category_name, all_categories)
Qwen_25_32b_metric = main(Qwen_25_32b_tree, Qwen_25_32b_LLM_conbined_small_model, task_to_category_name, all_categories)
Qwen_25_72b_metric = main(Qwen_25_72b_tree, Qwen_25_72b_LLM_conbined_small_model, task_to_category_name, all_categories)
Qwen_2_72b_metric = main(Qwen_2_72b_tree, Qwen_2_72b_LLM_conbined_small_model, task_to_category_name, all_categories)
Qwen_2_57b_metric = main(Qwen_2_57b_tree, Qwen_2_57b_LLM_conbined_small_model, task_to_category_name, all_categories)
Qwen_2_7b_metric = main(Qwen_2_7b_tree, Qwen_2_7b_LLM_conbined_small_model, task_to_category_name, all_categories)
llama_3_3_70b_metric = main(llama_3_3_70b_tree, llama_3_3_70b_LLM_conbined_small_model, task_to_category_name, all_categories)
gpt_4o_LLM_metric = main(gpt_4o_tree, gpt_4o_LLM_conbined_small_model, task_to_category_name, all_categories)
gpt_4o_mini_metric = main(gpt_4o_mini_tree, gpt_4o_mini_LLM_conbined_small_model, task_to_category_name, all_categories)
gpt_3_5_metric = main(gpt_3_5_tree, gpt_3_5_LLM_conbined_small_model, task_to_category_name, all_categories)
deepseek_2_5_metric = main(deepseek_2_5_tree, deepseek_2_5_LLM_conbined_small_model, task_to_category_name, all_categories)



print("--------------")
print("gpt-4o-metric")
print(gpt_4o_metric)
print("--------------")
print("gpt-4o-mini-metric")
print(gpt_4o_mini_metric)
print("--------------")
print("gpt-3.5-metric")
print(gpt_3_5_metric)
print("--------------")
print("deepseek_3_metric")
print(deepseek_3_metric)
print("--------------")
print("deepseek_2_5_metric")
print(deepseek_2_5_metric)
print("--------------")
print("llama_3_3_70B_metric")
print(llama_3_3_70B_metric)
print("--------------")
print("qwen_2_7B_metric")
print(Qwen_2_7b_metric)
print("--------------")
print("qwen_2_57B_metric")
print(Qwen_2_57b_metric)
print("--------------")
print("qwen_2_72B_metric")
print(Qwen_2_72b_metric)
print("--------------")
print("qwen_25_0.5B_metric")
print(Qwen_25_0_5b_metric)
print("--------------")
print("qwen_25_1.5B_metric")
print(Qwen_25_1_5b_metric)
print("--------------")
print("qwen_25_3B_metric")
print(Qwen_25_3b_metric)
print("--------------")
print("qwen_25_7B_metric")
print(Qwen_25_7b_metric)
print("--------------")
print("qwen_25_14B_metric")
print(Qwen_25_14b_metric)
print("--------------")
print("qwen_25_32B_metric")
print(Qwen_25_32b_metric)
print("--------------")
print("qwen_25_72B_metric")
print(qwen_25_72B_metric)



