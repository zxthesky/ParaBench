import json

def read_file(filename):
    if filename.endswith(".json"):
        with open(filename, 'r', encoding="utf-8") as f:
            all_data = json.load(f)
        return all_data
    elif filename.endswith(".jsonl"):
        all_data = []
        with open(filename, 'r', encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                all_data.append(data)
        return all_data
    else:
        if filename.endswith(".txt"):
            all_data = []
            with open(filename, 'r', encoding="utf-8")as f:
                for line in f:
                    all_data.append(json.loads(line))
            return all_data
        else:
            print("your filename is wrong")