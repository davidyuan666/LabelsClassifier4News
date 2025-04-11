

import os
import time
import numpy as np
import torch
import logging
from Config import Config
from DataManager import DataManager
from Trainer import Trainer
from Predictor import Predictor


'''
自己的数据集不需要做mask机制处理，代码会处理。
本项目目的在于基于现有的预训练模型参数，如google开源的bert-base-uncased、bert-base-chinese等，在垂直领域的数据语料上，再次进行预训练任务，由此提升bert的模型表征能力，换句话说，也就是提升下游任务的表现。
这个肯定不能重新开始训练的，其实finetune和pretrain本质上一样，只是下游任务的训练task不同，maskLM属于预训练的方式
python的版本为: 3.8
pip install datasets
pip install scikit-learn
https://pypi.tuna.tsinghua.edu.cn/simple (pip 源)
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
git clone https://www.modelscope.cn/AI-ModelScope/bert-base-uncased.git
git clone https://www.modelscope.cn/tiansz/bert-base-chinese.git
CUDA_VISIBLE_DEVICES=3 python main.py
'''
if __name__ == '__main__':


    config = Config()
    # os.environ["CUDA_VISIBLE_DEVICES"] = configs.cuda_visible_devices

    # 设置随机种子，保证结果每次结果一样
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    start_time = time.time()

    # 数据处理
    print('read data...')
    dm = DataManager(config)

    # 模式
    if config.mode == 'train':
        # 获取数据
        print('data process...')
        train_loader = dm.get_dataset(mode='train')
        valid_loader = dm.get_dataset(mode='dev')
        # 训练
        trainer = Trainer(config)
        trainer.train(train_loader, valid_loader)
    elif config.mode == 'test':
        # 测试
        test_loader = dm.get_dataset(mode='test', sampler=False)
        predictor = Predictor(config)
        predictor.predict(test_loader)
    else:
        print("no task going on!")
        print("you can use one of the following lists to replace the valible of Config.py. ['train', 'test', 'valid'] !")
        