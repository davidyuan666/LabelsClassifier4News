# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/23 19:33
@Auth ： David Yuan
@File ：bert_config.py
@Institute ：BitAuto
"""
import os
'''
hyperparameters
'''
class Config(object):
    '''
    BERT的配置
    '''
    MAX_LEN = 512  # 512 因为chinese_roberta_wwm_large_ext_pytorch只是优化了中文编码
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 16
    TEST_BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 1e-05
    dropout = 0.3
    # in_features = 768  # 这是默认的768的维度
    in_features = 1024  # 更新为1024，以匹配chinese_roberta_wwm_large_ext_pytorch模型，果然管用
    out_features = 10   # 这个必须和对应的类别数量对应起来
    # model_path = os.path.join(os.getcwd(), 'bert_pretrain_models','bert-base-chinese')
    model_path = os.path.join(os.getcwd(), 'bert_pretrain_models', 'chinese_roberta_wwm_large_ext_pytorch')
    # model_path = os.path.join(os.getcwd(), 'bert_pretrain_models', 'chinese-roberta-wwm-ext')
    '''
    TextCNN的配置
    '''
    hidden_size = 768
    filter_sizes = (6, 7, 8)  # 卷积核尺寸
    num_filters = 256  # 卷积核数量(channels数)
    dropout = 0.1
    num_classes = 13