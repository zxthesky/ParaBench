# import random

# a = ["1", "2", "3"]

# random.shuffle(a)
# print(a[2:])

gold_folder_name = "/data/xzhang/task_planning/main/data/new_main_task"

repeat_data = []

from utils import *

foldername = "/data/xzhang/task_planning/main/data/train_main_task"

all_filename = read_data_from_folder(foldername)

gold_filenames = read_data_from_folder(gold_folder_name)

print(len(all_filename))
print(len(gold_filenames))

gold_data_filename = "/data/xzhang/task_planning/main/data/final_data/test.json"

gold_test_data = read_file(gold_data_filename)

all_test_task_data = []

has_generate_data_filename = "/data/xzhang/task_planning/main/data/train_data/all_data_use_category.json"
has_generated_data = read_file(has_generate_data_filename)

print(len(has_generated_data))

final_has_generate_data = []

for data in has_generated_data:
    if data["task"] not in gold_test_data:
        final_has_generate_data.append(data)

print(len(final_has_generate_data))

