import json
import re


def read_file(filename):
    if filename.endswith(".json"):
        with open(filename, 'r') as f:
            all_data = json.load(f)
        return all_data
    elif filename.endswith(".jsonl"):
        all_data = []
        with open(filename, 'r') as f:
            for line in f:
                data = json.loads(line)
                all_data.append(data)
        return all_data
    else:
        print("your filename is wrong")

def write_file(all_data, filename):
    with open(filename, "w") as f:
        json.dump(all_data, f, indent=2)