from utils import *

'''获取一个数据集的情况'''

def get_subtasks(raw_data):
    subtasks = []

    for data in raw_data:
        now_subtask = data["step"]
        if now_subtask not in subtasks:
            subtasks.append(now_subtask)
    
    return subtasks
    

if __name__ == "__main__":
    
    filename = "/public/home/hmqiu/xzhang/task_planning/main/data/all_data_final/test.json"

    all_data = read_file(filename)
    print(len(all_data))

    compress_rate = 0
    average_length = 0
    average_depth = 0
    average_width = 0

    max_distance = 0

    max_length = 0
    max_width = 0

    max_length_reduce_depth = []
    for data in all_data:
        
        subtasks = get_subtasks(data["substeps"])
        dependencies = data["dependencies"]

        gold_answer_root_node = create_tree_with_subtasks(dependencies, subtasks)

        gold_depth =get_tree_depth(gold_answer_root_node)
        gold_width = get_tree_width(gold_answer_root_node)

        average_width += gold_width
        length = len(data["substeps"])
        depth = data["depth"]

        
        # print("***************************")
        # print(data["task"])
        # print(data["depth"])
        # print(gold_depth)
        # print(data["width"])
        # print(gold_width)

        average_length += length
        now_compress_rate = depth/length
        compress_rate += now_compress_rate
        average_depth += depth

        if max_length < length:
            max_length = length
        
        if max_width < gold_width:
            max_width = gold_width

       


    print(f"max_length : {max_length}")
    print(f"max_width : {max_width}")
    print(f"all_data_number : {len(all_data)}")
    print(f"average length : {average_length/len(all_data)}")
    print(f"average depth :  {average_depth/len(all_data)}")
    print(f"average width : {average_width/len(all_data)}")
    print(f"average compress rate :  {compress_rate/len(all_data)}")




