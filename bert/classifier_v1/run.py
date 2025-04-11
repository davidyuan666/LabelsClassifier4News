# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='BitAuto News Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()

'''
# 训练并测试：
# bert
CUDA_VISIBLE_DEVICES=0 python wisper_process.py --model bert

# bert + 其它
python wisper_process.py --model bert_CNN

# ERNIE
python wisper_process.py --model ERNIE
'''
if __name__ == '__main__':
    dataset = 'yichenews'  # 数据集

    '''
    选中模型
    '''
    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    '''
    划分数据集,build dataset 方法负责如何接受数据格式
    '''
    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    '''
    启动模型
    '''
    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)
