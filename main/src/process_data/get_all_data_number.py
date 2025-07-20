from utils import *


if __name__ == '__main__':
    filename = "/public/home/hmqiu/xzhang/task_planning/main/eval/eval_result_1005/qwen2.5_72b_instruct/tree.json"

    all_data = read_file(filename)
    print(len(all_data))
