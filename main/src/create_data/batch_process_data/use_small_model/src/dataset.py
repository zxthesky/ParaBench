from torch.utils.data import DataLoader, Dataset
import torch

from utils import read_file

class Tasklama(Dataset):
    """
    relation
    0 : 代表第一个任务要在第二个任务之前
    1 ：代表两个任务无关
    2 ： 代表第一个任务在第二个任务之后
    """
    def __init__(self, file_path, tokenizer, module="relation_train"):
        super(Tasklama, self).__init__()
        self.raw_data = read_file(file_path)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        
        now_data = self.raw_data[idx]

        task = now_data[0]

        former_sentence = now_data[1]
        later_sentence = now_data[2]
        temp_data_label = [0, 0, 0]
        label_index = now_data[3]
        temp_data_label[label_index] = 1.0
        temp_tokenizer = self.tokenizer(former_sentence, later_sentence, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        token_ids, attn_masks, token_type_ids = temp_tokenizer['input_ids'][0], temp_tokenizer['attention_mask'][0], temp_tokenizer['token_type_ids'][0]
        label = torch.tensor(temp_data_label)
        final_return = {}
        final_return["input_ids"] = token_ids
        final_return["attention_mask"] = attn_masks
        final_return["token_type_ids"] = token_type_ids
        final_return["labels"] = label

        return final_return
    
    def __len__(self):
        return len(self.raw_data)




