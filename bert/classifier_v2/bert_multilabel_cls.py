# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from config import Config

'''
BERT模型支持多标签输出
'''
class BertMultiLabelCls(nn.Module):
    def __init__(self, hidden_size, class_num):
        super(BertMultiLabelCls, self).__init__()
        self.bert = BertModel.from_pretrained(Config.pretrained_model_path)
        self.fc = nn.Linear(hidden_size, class_num)
        self.drop = nn.Dropout(Config.dropout)


    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        '''
        BertMultiLabelCls 使用 outputs[1]，这通常是 BERT 输出的 pooler_output，它是最后一个隐藏层状态的第一个令牌（通常是 [CLS] 令牌）的输出
        YicheBERT_Class 同样使用 output.pooler_output，这与 BertMultiLabelCls 的做法一致
        '''
        cls = self.drop(outputs[1])
        # out = F.sigmoid(self.fc(cls))
        '''
        BertMultiLabelCls 使用 torch.sigmoid 对最后的输出应用 Sigmoid 激活函数，这对于多标签分类任务是常见的做法。
        YicheBERT_Class 中没有显式使用激活函数，这可能意味着它用于不同类型的任务，或者激活函数的应用被留给了模型外部处理。
        '''
        out = torch.sigmoid(self.fc(cls))
        return out








