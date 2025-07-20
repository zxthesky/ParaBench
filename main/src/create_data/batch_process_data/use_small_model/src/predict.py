from utils import read_file
from bert_model import BertClassifier
from dataset import Tasklama
import torch
from torch import nn
from torch import optim as opt
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertForSequenceClassification,BertConfig
import argparse


def predict(model, predict_input):
    
    logits = model(**predict_input).logits

if __name__ == "__main__":
    model_path = ""
    model_parameter_path = ""
    bert_config = BertConfig.from_pretrained(model_path)
    model = BertForSequenceClassification(bert_config)
    state_dict = torch.load(model_parameter_path)
    model.load_state_dict(state_dict)
    bert_tokenizer = BertTokenizer.from_pretrained(model_path)

    data_file_path = ""

    raw_data = read_file(data_file_path)

    for data in raw_data:
        model.eval()
        edges = data["dependencies"]
        for edge in edges:
            former_sentence = edge["subtask1"]
            later_sentence = edge["subtask2"]
            temp_tokenizer = bert_tokenizer(former_sentence, later_sentence, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
            token_ids, attn_masks, token_type_ids = temp_tokenizer['input_ids'][0], temp_tokenizer['attention_mask'][0], temp_tokenizer['token_type_ids'][0]
            predict_input = {"input_ids": token_ids, "attention_mask": attn_masks, "token_type_ids": token_type_ids}
            predict(model, predict_input)




