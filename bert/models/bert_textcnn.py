# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/23 20:32
@Auth ： David Yuan
@File ：bert_model.py
@Institute ：BitAuto
"""
import torch
from transformers import BertModel
import torch.nn.functional as F
import os

'''
hyperparameters
'''
class Config(object):
    MAX_LEN = 512  # 512 因为chinese_roberta_wwm_large_ext_pytorch只是优化了中文编码
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 16
    TEST_BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 1e-05
    dropout = 0.3
    # in_features = 768  # 这是默认的768的维度
    in_features = 1024  # 更新为1024，以匹配chinese_roberta_wwm_large_ext_pytorch模型，果然管用
    model_path = os.path.join(os.getcwd(), 'bert_pretrain_models', 'chinese_roberta_wwm_large_ext_pytorch')
    
    # TextCNN相关参数
    hidden_size = 1024  # 与in_features保持一致
    filter_sizes = (2, 3, 4)  # 卷积核尺寸
    num_filters = 256  # 卷积核数量(channels数)
    num_classes = 0  # 使用时需要设置实际的类别数


'''
基础BERT分类模型
'''
class YicheBertModel(torch.nn.Module):
    def __init__(self, out_features):
        super(YicheBertModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(Config.model_path, return_dict=True)
        self.linear = torch.nn.Linear(Config.in_features, out_features)  # out_features就是最终输出的类别数量
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
升级版BERT+TextCNN模型
BERT搭配TextCNN，其中BERT主要提取文本特征，分类任务交给TextCNN完成
'''
class YicheBERT_TEXTCNN_Class(torch.nn.Module):
    def __init__(self, num_classes):
        super(YicheBERT_TEXTCNN_Class, self).__init__()
        # 设置类别数
        Config.num_classes = num_classes
        
        # 加载预训练BERT模型
        self.bert_model = BertModel.from_pretrained(Config.model_path, return_dict=True)
        
        # TextCNN部分
        # 为每个卷积核尺寸创建一个卷积层
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv2d(1, Config.num_filters, (k, Config.hidden_size)) for k in Config.filter_sizes])
        
        # Dropout层防止过拟合
        self.dropout = torch.nn.Dropout(Config.dropout)
        
        # 全连接层，将TextCNN提取的特征映射到类别空间
        self.fc_cnn = torch.nn.Linear(Config.num_filters * len(Config.filter_sizes), Config.num_classes)

    def conv_and_pool(self, x, conv):
        # 应用卷积并使用ReLU激活函数
        x = F.relu(conv(x)).squeeze(3)
        # 应用最大池化
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_ids, attn_mask, token_type_ids):
        # 通过BERT获取上下文表示
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
        )
        
        # 获取BERT的最后一层隐藏状态
        last_hidden_state = output.last_hidden_state

        # 调整维度以适应Conv2d层的输入要求 [batch_size, 1, seq_len, hidden_size]
        out = last_hidden_state.unsqueeze(1)

        # 对不同尺寸的卷积核的输出进行拼接
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        
        # 应用dropout
        out = self.dropout(out)
        
        # 通过全连接层输出分类结果
        out = self.fc_cnn(out)
        
        return out