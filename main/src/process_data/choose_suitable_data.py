from utils import *
import re
import json

filename = "/public/home/hmqiu/xzhang/task_planning/main/data/train_data_3000/train_data_3000.json"

filename_train_2000 = "/public/home/hmqiu/xzhang/task_planning/main/data/train_data_3000/train_data_2000.json"

train_2000 = read_file(filename_train_2000)

print(len(train_2000))

all_train_datas = read_file(filename)
print(len(all_train_datas))





def get_scores(raw_response):
    return re.findall(r"{\"score\": .*}", raw_response, re.DOTALL)[0]

def process_right_data_reduce_pass_and_score(all_true_datas):
    final_true_datas = []

    for data in all_true_datas:
        temp_data = {}
        temp_data["task"] = data["task"]
        temp_data["assumptions"] = data["assumptions"]
        temp_data["substeps"] = data["substeps"]
        temp_data["dependencies"] = data["dependencies"]
        temp_data["depth"] = data["depth"]
        temp_data["width"] = data["width"]
        final_true_datas.append(temp_data)
    return final_true_datas


def get_data_with_score_and_without(raw_data):
    final_datas_need_to_process = []
    right_datas = []
    for data in raw_data:
        if data.get("pass_or_not", -1) == -1:
            right_datas.append(data)
        else:
            final_datas_need_to_process.append(data)
    return final_datas_need_to_process, right_datas


def process_raw_data(raw_data):
    """根据每个数据的打分和通过率删掉一部分数据
    """

    final_datas_need_to_process, right_datas = get_data_with_score_and_without(raw_data)

    wrong_datas = []
    true_datas = []
    score_lower_than_5_datas = []
    for data in final_datas_need_to_process:
        raw_pass_or_not_data = data["pass_or_not"]
        raw_score_data = data["score"]
        score_str = get_scores(raw_score_data)
        score = json.loads(score_str)["score"]

        pass_or_not = False
        if "[yes]" in raw_pass_or_not_data.lower():
            pass_or_not = True
            true_datas.append(data)
        elif "[no]" in raw_pass_or_not_data.lower():
            wrong_datas.append(data)
        else:
            if score < 5:
                wrong_datas.append(data)
            else:
                true_datas.append(data)
        
        
        if score < 5:
            score_lower_than_5_datas.append(data)
    return true_datas, wrong_datas, right_datas


right_data_filename = "/public/home/hmqiu/xzhang/task_planning/main/data/train_data_3000/train_data.json"
wrong_data_filename = "/public/home/hmqiu/xzhang/task_planning/main/data/train_data_3000/final_wrong_data.json"
   
true_datas, wrong_datas, right_datas = process_raw_data(all_train_datas)

all_right_data = right_datas + true_datas 

now_right_data = true_datas + train_2000 
print("------------------")
print(len(now_right_data))
print(len(all_right_data))
print(len(true_datas))
print(len(wrong_datas))
print(len(right_datas))

# write_file(all_right_data, right_data_filename)
# write_file(wrong_datas, wrong_data_filename)
all_right_data_filename = "/public/home/hmqiu/xzhang/task_planning/main/data/train_data_3000/train_data_3000.json"
now_data = []
now_data = read_file(all_right_data_filename)
print(len(now_data))
# write_file(now_right_data, all_right_data_filename)
