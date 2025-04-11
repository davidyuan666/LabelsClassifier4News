# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import init_network,test,predict_and_save_to_csv
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

'''
CUDA_VISIBLE_DEVICES=3 python textcnn_run_test.py --model TextCNN   这个管用的
'''
if __name__ == '__main__':
    dataset = 'yiche-news-summary'   # 数据集

    '''
    载入词向量
    '''
    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('bert_pretrain_models.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    '''
    载入数据
    '''
    start_time = time.time()
    print("Loading test data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    '''
    载入测试数据，我们这里载入第一个textcnn去除二分类的部分
    '''
    dev_iter = build_iterator(dev_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    '''
    载入模型，并显示参数
    '''
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    '''
    开始测试
    
    配置曝光
    谍照
    实车曝光
    新车官图
    申报图
    新车到店

    '''
    test(config, model, dev_iter)
    # output_csv = 'predictions.csv'
    # predict_and_save_to_csv(configs,model,dev_iter, output_csv)