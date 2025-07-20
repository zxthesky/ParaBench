"""
去掉数据里面为了判别的pass_or_not和score标签
"""
from utils import *

filename = "/public/home/hmqiu/xzhang/task_planning/main/data/train_data/train_data_2037_has_judge.json"

raw_datas = read_file(filename)

write_filename = "/public/home/hmqiu/xzhang/task_planning/main/data/train_data/train_data.json"

final_datas = []


for data in raw_datas:
    temp_data = {}
    temp_data["task"] = data["task"]
    temp_data["assumptions"] = data["assumptions"]
    temp_data["substeps"] = data["substeps"]
    temp_data["dependencies"] = data["dependencies"]
    temp_data["depth"] = data["depth"]
    temp_data["width"] = data["width"]
    final_datas.append(data)

write_file(final_datas, write_filename)
