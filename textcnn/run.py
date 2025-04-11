# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse

'''
word参数默认为False，说明我们的中文文本是按单个字拆分训练的，True说明是按词语拆分
'''
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

'''
CUDA_VISIBLE_DEVICES=2 python run_bertrcnn.py --model TextCNN   这个管用的

CUDA_VISIBLE_DEVICES=3 python textcnn_run.py --model TextCNN

CUDA_VISIBLE_DEVICES=3 python wisper_process.py --model TextCNN


'''
if __name__ == '__main__':
    dataset_dir = 'yiche_news'  # 数据集
    '''
    载入词向量
    搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    '''
    model_name = 'TextCNN'  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    embedding_path = 'embedding_sougounews_p0.npz'
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding_path = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    '''
    导入models目录下的TextCNN文件
    '''
    x = import_module('models.' + model_name)
    '''
    加载TextCNN.py中的Config类，并初始化传入dataset和embedding,挤在textcnn对应的配置信息
    '''
    config = x.Config(dataset_dir, embedding_path)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    '''
    载入数据
    '''
    start_time = time.time()
    print("Loading data...")
    '''
    构建数据集以及载入iterator,目的是可以在有限的内存中加载数据，并支持batch数据，
    当然存在一种情况，就是batch中的负样本太少的话也会引起loss的抖动，这个部分还没有载入embedding
    '''
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)  #默认是False
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    '''
    载入模型
    '''
    config.n_vocab = len(vocab)
    '''
    这个Model就是对应文件的model的基类，将模型加载到GPU中
    '''
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        '''
        为什么要权重初始化而Transformer不用了
        '''
        init_network(model)

    print(model.parameters)
    '''
    开始训练
    '''
    train(config, model, train_iter, dev_iter, test_iter)
