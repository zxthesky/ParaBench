"""
这个文件用来对比 tree 和 tree_conbined_with_small_model
比较pass中相同任务的compress rate
"""
from utils import *

if __name__ == "__main__":
    tree_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2.5-14b-instruct/version_1_tree.json"
    tree_conbined_with_small_model_filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2.5-14b-instruct/version_1_LLM_conbine_small_model.json"

    tree_data = read_file(tree_filename)[:-1]
    tree_conbined_small_data = read_file(tree_conbined_with_small_model_filename)[:-1]

    tree_task_dict = {}

    for data in tree_data:
        if data.get("task", -1) == -1:
            print(data)
        now_key = data["task"] + str(data["subtasks"])
        tree_task_dict[now_key] = data


    final_tree_compression_rate = 0
    final_tree_small_compression_rate = 0

    all_same_pass_number = 0

    for data in tree_conbined_small_data:
        temp_key = data["task"] + str(data["subtasks"])
        if tree_task_dict.get(temp_key, -1) != -1:
            temp_tree_data = tree_task_dict[temp_key]
            temp_tree_combined_small_data = data
            a_whether_generate_wrong_subtask_judge = whether_generate_wrong_subtask(temp_tree_data["subtasks"], temp_tree_data["predict_dependencies"])
            b_whether_generate_wrong_subtask_judge = whether_generate_wrong_subtask(temp_tree_combined_small_data["subtasks"], temp_tree_combined_small_data["predict_dependencies"])

            if a_whether_generate_wrong_subtask_judge == False and b_whether_generate_wrong_subtask_judge == False:
                tree_gold_answer_root_node = create_tree_with_subtasks(temp_tree_data["gold_dependencies"], temp_tree_data["subtasks"])
                tree_predict_answer_root_node = create_tree_with_subtasks(temp_tree_data["predict_dependencies"], temp_tree_data["subtasks"])
                
                tree_small_gold_answer_root_node = create_tree_with_subtasks(temp_tree_combined_small_data["gold_dependencies"], temp_tree_combined_small_data["subtasks"])
                tree_small_predict_answer_root_node = create_tree_with_subtasks(temp_tree_combined_small_data["predict_dependencies"], temp_tree_combined_small_data["subtasks"])


                if tree_predict_answer_root_node != False and tree_small_predict_answer_root_node != False:
                    tree_raw_metric = compute_metric_one_data(tree_gold_answer_root_node, tree_predict_answer_root_node, temp_tree_data["subtasks"])
                    tree_small_metric = compute_metric_one_data(tree_small_gold_answer_root_node, tree_small_predict_answer_root_node, temp_tree_data["subtasks"])

                    tree_pass_rate = tree_raw_metric["pass_or_not"]
                    tree_small_pass_rate = tree_small_metric["pass_or_not"]

                    if tree_pass_rate == True and tree_small_pass_rate == True:
                        tree_compression_rate = tree_raw_metric["predict_depth"]/len(temp_tree_data["subtasks"])
                        tree_small_compress_rate = tree_small_metric["predict_depth"]/len(temp_tree_data["subtasks"])

                        final_tree_compression_rate += tree_compression_rate
                        final_tree_small_compression_rate += tree_small_compress_rate
                        all_same_pass_number += 1
    print(final_tree_compression_rate/all_same_pass_number)
    print(final_tree_small_compression_rate/all_same_pass_number)


