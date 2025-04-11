# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/23 15:10
@Auth ： David Yuan
@File ：bert_config.py
@Institute ：BitAuto
"""
import os
import torch
from data_preprocess import load_json

'''
Config 类已经是以静态方式定义了其属性，这意味着您可以直接通过类名访问这些属性
BERT multi-label configuration  BERT多标签分类的配置信息
'''
class Config(object):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_path = os.path.join(os.getcwd(),'data','configs.json')
    dev_path = os.path.join(os.getcwd(),'data','val.json')
    test_path = os.path.join(os.getcwd(),'data','test.json')
    label2idx_path = os.path.join(os.getcwd(),'data','label2idx.json')
    save_model_path = os.path.join(os.getcwd(),'checkpoint','multi_label_cls.pth')
    label2idx = load_json(label2idx_path)
    class_num = len(label2idx)
    pretrained_model_path = os.path.join(os.getcwd(),'model','bert-base-chinese')

    '''
    BERT模型的配置
    '''
    lr = 2e-5
    batch_size = 128
    max_len = 512
    hidden_size = 768
    epochs = 10
    dropout = 0.1