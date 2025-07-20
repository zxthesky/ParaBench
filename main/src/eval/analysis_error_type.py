"""
分析模型可能的错误类型，格式错误，未构成DAG，幻觉，未通过
"""
from utils import *

def eval_error_type(all_data: list, wrong_number: int=0):
    gold_wrong_node_task = []
    predict_wrong_node_task = []

    all_data = change_all_str_to_lower(all_data)


    ##############################

    formatted_error_number = wrong_number

    unable_create_tree = 0
    hallucination_number = 0

    un_pass_number = 0


    for data in all_data:
       
        whether_generate_wrong_subtask_judge = whether_generate_wrong_subtask(data["subtasks"], data["predict_dependencies"])
        if whether_generate_wrong_subtask_judge == False:
            gold_answer_root_node = create_tree_with_subtasks(data["gold_dependencies"], data["subtasks"], main_task=data["task"])
            predict_answer_root_node = create_tree_with_subtasks(data["predict_dependencies"], data["subtasks"])
            if gold_answer_root_node == False:
                continue
            if predict_answer_root_node != False: #### 确保生成了树
                
                raw_metric = compute_metric_one_data(gold_answer_root_node, predict_answer_root_node, data["subtasks"])

                pass_or_not = raw_metric["pass_or_not"]

                gold_wrong_node_information = raw_metric["gold_wrong_node_information"]
                predict_wrong_node_information = raw_metric["predict_wrong_node_task"]
                gold_now_wrong_node_information = {"task": data["task"], "wrong_node_information": gold_wrong_node_information}
                predict_now_wrong_node_information = {"task": data["task"], "wrong_node_information": predict_wrong_node_information}


                
                gold_wrong_node_task.append(gold_now_wrong_node_information)
                predict_wrong_node_task.append(predict_now_wrong_node_information)

                ###################################
                


                if pass_or_not == True:
                    pass

                else:
                    un_pass_number += 1

            else:
                un_pass_number += 1
                unable_create_tree += 1
        else:
            un_pass_number += 1
            hallucination_number += 1

    un_pass_number += wrong_number
    task_error_number = un_pass_number - formatted_error_number - unable_create_tree - hallucination_number

    print("****************************")
    print(un_pass_number)
    print(formatted_error_number)
    print(unable_create_tree)
    print(hallucination_number)


    formatted_rate = formatted_error_number/un_pass_number
    uncreate_tree_rate = unable_create_tree/un_pass_number
    hallucination_rate = hallucination_number/un_pass_number
    task_error_rate = task_error_number/un_pass_number


    return {"Format Error": formatted_rate, "Non-DAG Structure": uncreate_tree_rate, "Hallucination_rate": hallucination_rate, "Planning Error": task_error_rate}

        


def main(gold_data_filename:str, filename:str):
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


    result = eval_error_type(model_predict_data, wrong_number=wrong_number)


    print(result)
    print("***********************")


if __name__ == '__main__':
    gold_data_filename = "/public/home/hmqiu/xzhang/task_planning/main/data/final_data/test.json"
    filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2.5-0.5b-instruct/tree.json"

    main(gold_data_filename, filename)

