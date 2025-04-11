# -*- coding: utf-8 -*-
'''
author: David Yuan
institution: BitAutoTechnologies, Inc
'''
import os.path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
import numpy as np
from bert_multilabel_cls import BertMultiLabelCls
from data_helper import MultiClsDataSet
from config import Config  # 导入Config,直接用静态的方法进行访问
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from data_preprocess import preprocess
import os
import random
import json

def get_acc_score(y_true_tensor, y_pred_tensor):
    y_pred_tensor = (y_pred_tensor.cpu() > 0.5).int().numpy()
    y_true_tensor = y_true_tensor.cpu().numpy()
    return accuracy_score(y_true_tensor, y_pred_tensor)

'''
提前运行data_preprocess.py获取label2id,也就是标签id
'''
def train():
    # 需要提前获取label2idx_path
    train_dataset = MultiClsDataSet(Config.train_path, max_len=Config.max_len)
    dev_dataset = MultiClsDataSet(Config.dev_path, max_len=Config.max_len)
    '''
    数据集加载
    '''
    train_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=Config.batch_size, shuffle=False)

    '''
    加载模型，自定义的多标签模型
    '''
    model = BertMultiLabelCls(hidden_size=Config.hidden_size, class_num=Config.class_num)
    model.to(Config.device)
    model.train()

    '''
    优化器，和loss类型
    '''
    optimizer = AdamW(model.parameters(), lr=Config.lr)
    criterion = nn.BCELoss()

    dev_best_acc = 0.

    for epoch in range(1, Config.epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = [d.to(Config.device) for d in batch]
            labels = batch[-1]
            logits = model(*batch[:3])
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if i % 5 == 0:
                acc_score = get_acc_score(labels, logits)
                print("Train epoch:{} step:{}  acc: {} loss:{} ".format(epoch, i, acc_score, loss.item()))

        # 验证集合
        dev_loss, dev_acc = dev(model, dev_dataloader, criterion)
        print("Dev epoch:{} acc:{} loss:{}".format(epoch, dev_acc, dev_loss))
        if dev_acc > dev_best_acc:
            dev_best_acc = dev_acc
            torch.save(model.state_dict(), Config.save_model_path)

    # 测试
    test_acc_score,test_precision, test_recall, test_f1, test_macro_precision, test_macro_recall, test_macro_f1 = test(Config.save_model_path, Config.test_path)
    # 格式化输出
    print("Test Accuracy: {:.4f}".format(test_acc_score))
    print("Test Precision: {:.4f}".format(test_precision))
    print("Test Recall: {:.4f}".format(test_recall))
    print("Test F1 Score: {:.4f}".format(test_f1))
    print("Test Macro Precision: {:.4f}".format(test_macro_precision))
    print("Test Macro Recall: {:.4f}".format(test_macro_recall))
    print("Test Macro F1 Score: {:.4f}".format(test_macro_f1))


def dev(model, dataloader, criterion):
    all_loss = []
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids, attention_mask, token_type_ids, labels = [d.to(Config.device) for d in batch]
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = criterion(logits, labels)
            all_loss.append(loss.item())
            true_labels.append(labels)
            pred_labels.append(logits)
    true_labels = torch.cat(true_labels, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    acc_score = get_acc_score(true_labels, pred_labels)
    return np.mean(all_loss), acc_score


def test(model_path, test_data_path):
    test_dataset = MultiClsDataSet(test_data_path, max_len=Config.max_len, label2idx_path=Config.label2idx_path)
    test_dataloader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)
    model = BertMultiLabelCls(hidden_size=Config.hidden_size, class_num=Config.class_num)
    model.load_state_dict(torch.load(model_path))
    model.to(Config.device)
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            input_ids, attention_mask, token_type_ids, labels = [d.to(Config.device) for d in batch]
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            true_labels.append(labels)
            pred_labels.append(logits)
    true_labels = torch.cat(true_labels, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    acc_score = get_acc_score(true_labels, pred_labels)
    precision, recall, f1, macro_precision, macro_recall, macro_f1 = get_metrics(true_labels,pred_labels)
    return acc_score,precision, recall, f1, macro_precision, macro_recall, macro_f1



def get_metrics(true_labels, pred_labels):
    pred_labels = torch.sigmoid(pred_labels).cpu().numpy()
    pred_labels = np.where(pred_labels > 0.5, 1, 0)
    true_labels = true_labels.cpu().numpy()

    precision = precision_score(true_labels, pred_labels, average=None)
    recall = recall_score(true_labels, pred_labels, average=None)
    f1 = f1_score(true_labels, pred_labels, average=None)

    macro_precision = precision_score(true_labels, pred_labels, average='macro')
    macro_recall = recall_score(true_labels, pred_labels, average='macro')
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')

    return precision, recall, f1, macro_precision, macro_recall, macro_f1

def split_json_data(input_file, train_file, val_file, test_file, train_ratio=0.8, val_ratio=0.1):
    if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
        print("Train, validation, and test files already exist. Skipping split.")
        return

    with open(input_file, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]

    random.shuffle(data)

    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    test_size = len(data) - train_size - val_size

    train_data = data[:train_size]
    val_data = data[train_size:(train_size + val_size)]
    test_data = data[-test_size:]

    with open(train_file, 'w', encoding='utf-8') as file:
        for item in train_data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_file, 'w', encoding='utf-8') as file:
        for item in val_data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(test_file, 'w', encoding='utf-8') as file:
        for item in test_data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")




def start():
    print('查看是否需要生成label2idx.json文件')
    if not os.path.exists(os.path.join(os.getcwd(),'data','label2idx.json')):
        print('开始生成')
        train_path = os.path.join(os.getcwd(), 'data', 'all_multi_label.json')
        label2idx_path = os.path.join(os.getcwd(), 'data', 'label2idx.json')
        preprocess(train_path, label2idx_path)
        print('生成完毕')
    else:
        print('不需要生成label2idx.json文件')

    print('查看是否需要划分训练集')
    input_file = os.path.join(os.getcwd(),'data','all_multi_label.json')
    train_file = os.path.join(os.getcwd(),'data','configs.json')
    val_file = os.path.join(os.getcwd(),'data','val.json')
    test_file = os.path.join(os.getcwd(),'data','test.json')
    split_json_data(input_file, train_file, val_file, test_file)
    print('划分完毕')
    print('开始训练模型.....')
    os.environ['CUDA_VISIBLE_DEVICES'] = Config.device
    train()


'''
这个版本的BERT的最后激活函数部分有点不同
transformers=4.18.0
CUDA_VISIBLE_DEVICES=3 python train_bert_textcnn.py
'''
if __name__ == '__main__':
    start()
