"""
此文件读取模型生成文件，不需要跑模型，之间预测
"""
from utils import *

def main(gold_data_filename:str, filename:str, gold_wrong_write_filename:str, predict_wrong_write_filename:str):
    """
    Args:
        gold_data_filename(str): 原始test的金标数据
        filename(str): 模型预测数据
        gold_wrong_write_filename(str): 当前数据中有哪些依赖并没有满足，每个主任务task对应的规划有哪些边没用满足。(金标边里面没有满足的)
        predict_wrong_write_filename(str): 当前数据中有哪些依赖并没有满足，每个主任务task对应的规划有哪些边没用满足。(预测的边中有哪些不符合金标的关系)
    """
    raw_gold_data = read_file(gold_data_filename)
    test_number = len(raw_gold_data)

    model_datas = read_file(filename)
    print(len(model_datas))
    metric = model_datas[-1]
    
    if "metric" in list(metric.keys()):
        model_predict_data = model_datas[:-1]
    else:
        model_predict_data = model_datas

   
    print(len(model_predict_data))

    model_predict_data_number = len(model_predict_data)

    wrong_number = test_number - model_predict_data_number

    # model_predict_data = model_predict_data[:1]
    # print(model_predict_data[0]["task"])

    result, gold_wrong_node_task, predict_wrong_node_task= eval_all_data(model_predict_data, wrong_number=wrong_number)
    # result = eval_all_data(model_predict_data, wrong_number=wrong_number)
    # write_file(gold_wrong_node_task, gold_wrong_write_filename)
    # write_file(predict_wrong_node_task, predict_wrong_write_filename)

    print(result)
    print("***********************")


if __name__ == '__main__':
    gold_data_filename = "/public/home/hmqiu/xzhang/task_planning/main/data/final_data/test.json"
    filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/DeepSeek-V2.5/linear.json"
    gold_wrong_write_filename = "/data/xzhang/task_planning/main/eval/model_wrong_node_information/llama3-instruct/llama3_tree_gold__wrong_dependencies.json"
    predict_wrong_write_filename = "/data/xzhang/task_planning/main/eval/model_wrong_node_information/llama3-instruct/llama3_tree_predict_wrong_dependencies.json"
    main(gold_data_filename, filename, gold_wrong_write_filename, predict_wrong_write_filename)

