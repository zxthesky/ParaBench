from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertForSequenceClassification,BertConfig

model_path = "/data/xzhang/model_parameter/bert/bert_based_uncased"

mytokenizer = BertTokenizer.from_pretrained(model_path)

task = "build a house"
former_sentence = "bug ingredients"
later_sentence = "install windows"
former_sentence_token = mytokenizer(former_sentence, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
later_sentence_token = mytokenizer(later_sentence, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
temp_tokenizer = mytokenizer(former_sentence, later_sentence, padding='max_length', truncation=True, max_length=32, return_tensors='pt')

all_tokens = mytokenizer(task, former_sentence, later_sentence, padding='max_length', truncation=True, max_length=32, return_tensors='pt')

print(former_sentence_token)
print(later_sentence_token['input_ids'])
print(temp_tokenizer['input_ids'])
print(all_tokens['input_ids'])

