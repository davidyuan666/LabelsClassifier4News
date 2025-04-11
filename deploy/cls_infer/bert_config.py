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
    MAX_LEN = 512  # 512 因为chinese_roberta_wwm_large_ext_pytorch只是优化了中文编码
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 16
    TEST_BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 1e-05
    dropout = 0.3
    # in_features = 768  # 这是默认的768的维度
    in_features = 1024  # 更新为1024，以匹配chinese_roberta_wwm_large_ext_pytorch模型，果然管用
    base_www_bert_model_path = '/data/sharedpvc/david_tag_cls'
    model_path = os.path.join(base_www_bert_model_path, 'bert_pretrain_models', 'chinese_roberta_wwm_large_ext_pytorch')