# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/25 11:48
@Auth ： David Yuan
@File ：predict_p0.py
@Institute ：BitAuto
"""

import pandas as pd
import shutil
from transformers import BertTokenizer
import os
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import torch
from config import Config
from dataset_loader import YicheNewsDataset
from bert_model import YicheBERT_Class,YicheBERT_TEXTCNN_Class


all_data_csv_path = 'data/20230101_news_and_yichehao_news_no_zero_rows_p1.csv'
all_df = pd.read_csv(all_data_csv_path)
all_df.info()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained(Config.model_path)

# target_list = ['谍照','实车曝光','配置曝光','申报图','预热','新车上市','预售','发布亮相','新车官图','新车报价','新车到店','新车解析']
target_list = ['车系品牌解读', '单车导购', '对比导购', '多车导购', '购车手册', '买车技巧', '评测导购','汽车分享', '试驾', '探店报价', '无解说车辆展示', '营销导购']

train_size = 0.7
test_size = 0.2  # 剩余的0.1将用于验证集
train_df = all_df.sample(frac=train_size, random_state=200).reset_index(drop=True)
remaining_df = all_df.drop(train_df.index).reset_index(drop=True)
test_df = remaining_df.sample(frac=(test_size / (1 - train_size)), random_state=200).reset_index(drop=True)
val_df = remaining_df.drop(test_df.index).reset_index(drop=True)


train_dataset = YicheNewsDataset(train_df, tokenizer, Config.MAX_LEN, target_list)
valid_dataset = YicheNewsDataset(val_df, tokenizer, Config.MAX_LEN, target_list)
test_dataset = YicheNewsDataset(test_df, tokenizer, Config.MAX_LEN, target_list)

train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=Config.TRAIN_BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=0
                                                )

val_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                              batch_size=Config.VALID_BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=0
                                              )

test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=Config.TEST_BATCH_SIZE,
                                               shuffle=False, num_workers=0
                                               )


model = YicheBERT_Class()
model.to(device)
optimizer = torch.optim.Adam(params = model.parameters(), lr=Config.LEARNING_RATE)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_loss_min = checkpoint['valid_loss_min']
    return model, optimizer
    # return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def convert_to_labels(label_list, label_indices):
    return [label for label, index in zip(label_list, label_indices) if index == 1]


def predict(model, data_loader, label_list):
    import csv
    import time
    threshold = 0.5
    model.eval()
    all_records = []

    with torch.no_grad():
        for data in data_loader:
            # 在预测后打印显存信息
            print_cuda_memory_usage('预测时')
            start_time = time.time()
            content_ids = data['content_ids']
            contents = data['contents']
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)
            outputs = torch.sigmoid(outputs).cpu().detach().numpy() > threshold

            # 收集每个样本的原始内容、真实标签和预测标签
            for content_id,content, target, output in zip(content_ids,contents, targets, outputs):
                true_labels = convert_to_labels(label_list, target)
                predicted_labels = convert_to_labels(label_list, output)
                true_labels_str = ','.join(true_labels)
                predicted_labels_str = ','.join(predicted_labels)
                all_records.append((content_id,content, true_labels_str, predicted_labels_str))

            end_time = time.time()
            print(f'batch size: {Config.TEST_BATCH_SIZE} cost time: {end_time-start_time} seconds')


    # 写入 CSV 文件
    # with open('p1_bert_wwmlarge_prediction_own.csv', 'w', newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['ContentID','Content', 'True Labels', 'Predicted Labels'])
    #
    #     for record in all_records:
    #         content_id,content, true_labels_str, predicted_labels_str = record
    #         writer.writerow([content_id,content, true_labels_str, predicted_labels_str])



def print_cuda_memory_usage(note=''):
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB单位
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB单位
    print(f"{note} | 已分配显存: {allocated:.2f} GB, 历史最高分配显存: {max_allocated:.2f} GB")



'''
CUDA_VISIBLE_DEVICES=1 python predict_p1.py    
'''
if __name__ == '__main__':
    '''
    加载模型，并预测结果保存在文件中,测试结果用测试集
    如果模型已经初始化,只需要加载参数就行
    '''
    # 在预测前打印显存信息
    print_cuda_memory_usage('预测前')
    checkpoint_model_path = os.path.join(os.getcwd(), 'save_bert_wwmlarge_models_p1', 'curr_ckpt', 'checkpoint_epoch_6.pt')
    checkpoint_model, optimizer = load_ckp(checkpoint_model_path, model, optimizer)

    label_list = ['车系品牌解读', '单车导购', '对比导购', '多车导购', '购车手册', '买车技巧', '评测导购', '汽车分享',
                   '试驾', '探店报价', '无解说车辆展示', '营销导购']

    print(f'开始预测')
    predict(model, test_data_loader, label_list)



