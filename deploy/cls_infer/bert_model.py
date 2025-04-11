# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/23 20:32
@Auth ： David Yuan
@File ：bert_model.py
@Institute ：BitAuto
"""
import torch
from transformers import BertModel
from bert_config import Config
import torch.nn.functional as F

'''

在BERT（或类似的预训练模型）中使用自定义预训练的词嵌入（embeddings）是可能的，但它需要一些调整和考虑。
通常，BERT模型自带的词嵌入已经通过大量数据进行了预训练，因此它们已经非常有效。但如果您有特定领域的词嵌入，
这些嵌入可能会对模型性能产生正面影响。
模型搭建过程，在bert模型基础上搭建了其他外围的组件
'''
class YicheBertModel(torch.nn.Module):
    def __init__(self,out_features):
        super(YicheBertModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(Config.model_path, return_dict=True)
        self.linear = torch.nn.Linear(Config.in_features, out_features)  #out_featues就是最终输出的类别数量
        self.dropout = torch.nn.Dropout(Config.dropout)


    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )
        '''
        BertMultiLabelCls 使用 outputs[1]，这通常是 BERT 输出的 pooler_output，它是最后一个隐藏层状态的第一个令牌（通常是 [CLS] 令牌）的输出
        YicheBERT_Class 同样使用 output.pooler_output，这与 BertMultiLabelCls 的做法一致
        '''
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output


'''
升级版
BERT搭配textcnn，其中BERT主要做特征，分类交给textcnn
'''
class YicheBERT_TEXTCNN_Class(torch.nn.Module):
    def __init__(self):
        super(YicheBERT_TEXTCNN_Class, self).__init__()
        self.bert_model = BertModel.from_pretrained(Config.model_path, return_dict=True)

        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv2d(1, Config.num_filters, (k, Config.hidden_size)) for k in Config.filter_sizes])
        self.dropout = torch.nn.Dropout(Config.dropout)
        self.fc_cnn = torch.nn.Linear(Config.num_filters * len(Config.filter_sizes), Config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # Apply convolution and then ReLU
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # Apply max pooling
        return x

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids)
        last_hidden_state = output.last_hidden_state

        # 调整 last_hidden_state 维度以匹配 Conv2d 层的输入要求
        out = last_hidden_state.unsqueeze(1)  # [batch_size, 1, sequence_length, hidden_size]

        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out
