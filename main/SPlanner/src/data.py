import torch
from .utils import *
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class TaskPlanner(Dataset):
    def __init__(self, filepath, tokenizer, max_seq_len=512):
        super(TaskPlanner, self).__init__()
        self.max_seq_len = max_seq_len
        self.step_token = "[step]"
        self.assumption_token = "[assumption]"
        self.tokenizer = tokenizer
        self.raw_datas = read_file(filepath)
        

    def __getitem__(self, index):
        return self.process_instance(self.raw_datas[index])

    def __len__(self):
        return len(self.raw_datas)

    def process_instance(self, ins):
        substeps = sorted(ins["substeps"], key=lambda x: x["stepId"])
        assumptions = sorted(ins["assumptions"], key=lambda x: x["assumptionId"])
        substeps = [s["step"] for s in substeps]
        assumptions = [a["assumption"] for a in assumptions]

        step2id = {s: i for i, s in enumerate(substeps)}
        dep_pairs = []
        for dep in ins["dependencies"]:
            dep_pairs.append((step2id[dep["subtask1"]], step2id[dep["subtask2"]]))
            

        tokenized = {
            "task": self.tokenizer.tokenize(ins["task"], add_special_tokens=False),
            "assumptions": [
                self.tokenizer.tokenize(assumption, add_special_tokens=False)
                for assumption in assumptions
            ],
            "substeps": [
                self.tokenizer.tokenize(substep, add_special_tokens=False)
                for substep in substeps
            ],
        }

        input_tokens = [self.tokenizer.cls_token]
        mask = [1]
        input_tokens += tokenized["task"]
        mask += [2] * len(tokenized["task"])
        for assumption in tokenized["assumptions"]:
            input_tokens += [self.assumption_token]
            mask += [4]
            input_tokens += assumption
            mask += [5] * len(assumption)
        step_token_pos = []
        for step in tokenized["substeps"]:
            step_token_pos.append(len(input_tokens))
            input_tokens += [self.step_token]
            mask += [6]
            input_tokens += step
            mask += [7] * len(step)
        input_tokens += [self.tokenizer.sep_token]
        mask += [8]

        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        label_positions = []
        for pair in dep_pairs:
            label_positions.append((step_token_pos[pair[0]], step_token_pos[pair[1]]))

        res = {
            "input_tokens": input_tokens,
            "input_ids": input_ids,
            "mask": mask,
            "labels": label_positions,
            "raw_data": ins
        }

        return res
    
    @staticmethod
    def pad(seq, pad: int = 0, max_len: int = None):
        if max_len is None:
            max_len = max([len(s) for s in seq])
        padded = []
        for s in seq:
            padded.append(s[:max_len] + [pad] * (max_len - len(s)))
        return padded

    def collate_fn(self, features):
        seq_lens = [len(ft["input_ids"]) for ft in features]
        input_ids = torch.tensor(
            self.pad(
                [ft["input_ids"] for ft in features],
                pad=self.tokenizer.pad_token_id,
                max_len=self.max_seq_len,
            ),
            dtype=torch.long,
        )
        mask = torch.tensor(
            self.pad(
                [ft["mask"] for ft in features],
                pad=self.tokenizer.pad_token_id,
                max_len=self.max_seq_len,
            ),
            dtype=torch.long,
        )
        curr_max_seq_len = input_ids.shape[1]
        labels = torch.zeros(
            len(features), curr_max_seq_len, curr_max_seq_len, dtype=torch.long
        )
        for i, (ins_len, ft) in enumerate(zip(seq_lens, features)):
            # pad
            
            labels[i, :ins_len, :ins_len] = -100
            
            # 6 is the mask of step token
            step_label_mask = (mask[i] == 6).float().expand_as(labels[i])
            
            step_label_mask = step_label_mask * step_label_mask.t()
            
            labels[i][step_label_mask == 0] = -100
            
            labels[i][step_label_mask == 1] = 0

            
            
            # normal labels
            for pair in ft["labels"]:
                labels[i, pair[0], pair[1]] = 1

        return {"input_ids": input_ids, "mask": mask, "labels": labels}

            
if __name__ == "__main__":
    filename = "/data/xzhang/task_planning/main/data/test_data/train.json"
    tokenizer_dir = "/data/xzhang/model_parameter/deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, legacy=False, use_fast=True)
    train_dataset = TaskPlanner(filename, tokenizer)
    train_loader = DataLoader(train_dataset)
    for train_data in train_loader:
        
        print(train_data.keys())
        

