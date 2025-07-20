from utils import read_file
from dataset import Tasklama
import torch
from torch import nn
from torch import optim as opt
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup,Trainer, BertTokenizer, BertForSequenceClassification,BertConfig
import argparse
from tqdm import *
import numpy as np



def valiation(model, dev_loader, device, loss_fn):
    total_loss = 0
    step_num = 0
    model.eval()
    with torch.no_grad():
        for input in dev_loader:
            step_num += 1
            input = {k: v.to(device) for k,v in input.items()}
            output = model(**input)
            loss = output.loss
            total_loss += loss
        average_loss = total_loss/step_num
    print(f"'--eval--'  average loss: {average_loss}")

def test(model, test_loader, device):
    all_number = 0
    right_number = 0
    model.eval()
    with torch.no_grad():
        for input in test_loader:
            input = {k: v.to(device) for k,v in input.items()}
            output = model(**input)
            logits = output.logits
            pre = logits.argmax(1)
            label = input["labels"].argmax(1)
            all_number += len(pre)
            for i in range(len(pre)):
                if pre[i] == label[i]:
                    right_number += 1
    print("-------------test result------------------")
    print(f"--------------{right_number/all_number}-----------------")

           


def train(train_data_path, dev_data_path, test_data_path, model_path, epoch, max_len, save_model_path, batch_size=2,lr=1e-5,eps=1e-8):
    """
    使用transformer的库
    """
    # config = BertConfig.from_pretrained(model_path)
    
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
    mytokenizer = BertTokenizer.from_pretrained(model_path)
    train_dataset = Tasklama(train_data_path, mytokenizer)
    dev_dataset = Tasklama(dev_data_path, mytokenizer)
    test_dataset = Tasklama(test_data_path, mytokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(),lr=lr, eps=eps)
    best_loss = 1000

    print("-----------------starting training ----------------------")

    for i in tqdm(range(epoch)):
        loss = 10
        step_num = 0 
        model.train()
        
        for input in train_loader:
            step_num += 1
            input = {k: v.to(device) for k,v in input.items()}
            output = model(**input)
            loss = output.loss
            loss.backward()
            optimizer.step()
            
            logits = output.logits

            if step_num % 1000 == 0:
                print(f"step: {step_num}; loss: {loss}")
            optimizer.zero_grad()

        print(f"epoch: {i}; loss: {loss}")
        if loss < best_loss:
            torch.save(model, save_model_path)
        valiation(model, dev_loader, device, loss_fn)    

    test(model, test_loader, device) 




if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, required=True, help="the path of model parameter")
    # parser.add_argument("--output_answer_file", type=str, required=True, help="output file path")
    # parser.add_argument("--train_data_path", type=str, required=True, help="the data path")
    # parser.add_argument("--dev_data_path", type=str, required=True, help="the data path")
    # parser.add_argument("--epoch", type=int, default=30, required=False, help="the data path")
    # parser.add_argument("--max_len", type=int, default=16, required=False, help="the data path")
    # parser.add_argument("--save_model_path", type=str, required=True, help="train model saved path")

    # args = parser.parse_args()
    # train_data_path = args.train_data_path
    # dev_data_path = args.dev_data_path
    # epoch = args.epoch
    # model_path = args.model_path
    # max_len = args.max_len
    # save_model_path = args.save_model_path
    
    model_path = "/data/xzhang/model_parameter/bert/bert_based_uncased"
    train_data_path = "/data/xzhang/task_planning/main/create_data/batch_process_data/use_small_model/data/train.json"
    dev_data_path = "/data/xzhang/task_planning/main/create_data/batch_process_data/use_small_model/data/dev.json"
    test_data_path = "/data/xzhang/task_planning/main/create_data/batch_process_data/use_small_model/data/test.json"

    epoch = 20
    max_len = 128
    save_model_path = "/data/xzhang/task_planning/main/create_data/batch_process_data/use_small_model/save_model_parameter/bert_classsfication_model_save.bin"

    train(train_data_path, dev_data_path, test_data_path, model_path, epoch, max_len, save_model_path)







