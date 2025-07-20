from utils import read_file, write_file
import random

def process_dependency_data(raw_data):
    final_data_we_need = []
    for data in raw_data:
        task = data["task"]
        for edge in data["data"]:
            subtask1 = edge["subtask1"]
            subtask2 = edge["subtask2"]
            final_data_we_need.append([task, subtask1, subtask2, 0])
            final_data_we_need.append([task, subtask2, subtask1, 2])
    return final_data_we_need


def process_unrelevant_data(raw_data):
    final_data_we_need = []
    for data in raw_data:
        task = data["task"]
        for edge in data["data"]:
            subtask1 = edge["subtask1"]
            subtask2 = edge["subtask2"]
            final_data_we_need.append([task, subtask1, subtask2, 1])
            final_data_we_need.append([task, subtask2, subtask1, 1])
    return final_data_we_need


if __name__ == "__main__":

    train_data_filename = "/data/xzhang/task_planning/main/create_data/batch_process_data/use_small_model/data/train.json"
    dev_data_filename = "/data/xzhang/task_planning/main/create_data/batch_process_data/use_small_model/data/dev.json"
    test_data_filename = "/data/xzhang/task_planning/main/create_data/batch_process_data/use_small_model/data/test.json"

    dependencies_filename = "/data/xzhang/task_planning/main/create_data/batch_process_data/use_small_model/dependencies_data/dependencies.json"
    unrelevant_filename = "/data/xzhang/task_planning/main/create_data/batch_process_data/use_small_model/unrelevants_data/unrelevants.json"

    dependencies_data = process_dependency_data(read_file(dependencies_filename))
    unrelevant_data = process_unrelevant_data(read_file(unrelevant_filename))
    raw_data = dependencies_data + unrelevant_data

    random.shuffle(raw_data)

    all_data_number = len(raw_data)
    train_limit = int(0.8*all_data_number)
    dev_limit = int(0.9*all_data_number)
    print(dev_limit)
    train_data = raw_data[:train_limit]
    dev_data = raw_data[train_limit:dev_limit]
    test_data = raw_data[dev_limit:]

    write_file(train_data, train_data_filename)
    write_file(dev_data, dev_data_filename)
    write_file(test_data, test_data_filename)

