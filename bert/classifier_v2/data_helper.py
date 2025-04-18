# -*- coding: utf-8 -*-
import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from data_preprocess import load_json

from config import Config
'''
数据集类
'''
class MultiClsDataSet(Dataset):
    def __init__(self, data_path, max_len=128):
        self.label2idx = load_json(Config.label2idx_path)  #把label转为id表示
        self.class_num = len(self.label2idx)  #可以获取到label的种类数量
        self.tokenizer = BertTokenizer.from_pretrained(Config.pretrained_model_path)
        self.max_len = max_len
        self.input_ids, self.token_type_ids, self.attention_mask, self.labels = self.encoder(data_path)


    '''
    对字符进行分词和转为id
    '''
    def encoder(self, data_path):
        texts = []
        labels = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                texts.append(line["text"])
                tmp_label = [0] * self.class_num
                for label in line["label"]:
                    tmp_label[self.label2idx[label]] = 1
                labels.append(tmp_label)

        tokenizers = self.tokenizer(texts,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_len,
                                    return_tensors="pt",
                                    is_split_into_words=False)
        input_ids = tokenizers["input_ids"]
        token_type_ids = tokenizers["token_type_ids"]
        attention_mask = tokenizers["attention_mask"]

        return input_ids, token_type_ids, attention_mask, \
               torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.input_ids[item],  self.attention_mask[item], \
               self.token_type_ids[item], self.labels[item]
