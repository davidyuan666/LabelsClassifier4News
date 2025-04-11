# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/8 09:54
@Auth ： David Yuan
@File ：train_bert_textcnn.py
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
from dataset_loader import YicheVideosDataset
from bert_model import YicheBERT_Class,YicheBERT_TEXTCNN_Class

'''
load the textcnn_dataset
多标签的多标签编码
类似 0,0,0,0,0,1,0,0,0,1,0,1,0
'''
all_data_csv_path = 'data/yiche_video_news_p0.csv'
# all_data_csv_path = 'data/20230101_news_and_yichehao_news_p1.csv'
all_df = pd.read_csv(all_data_csv_path)
all_df.info()

'''
下载tokenizer器,使用传统的试试,加载本地模型
'''
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained(Config.model_path)

'''
column的名称,这个很重要，主要是为了和多标签编码进行映射
'''
target_list = ['谍照','实车曝光','配置曝光','申报图','预热','新车上市','预售','发布亮相','新车官图','新车报价','新车到店','新车解析']

train_size = 0.7
test_size = 0.2  # 剩余的0.1将用于验证集
train_df = all_df.sample(frac=train_size, random_state=200).reset_index(drop=True)
remaining_df = all_df.drop(train_df.index).reset_index(drop=True)
test_df = remaining_df.sample(frac=(test_size / (1 - train_size)), random_state=200).reset_index(drop=True)
val_df = remaining_df.drop(test_df.index).reset_index(drop=True)


'''
把读取的df文件载入dataset中
'''
train_dataset = YicheVideosDataset(train_df, tokenizer, Config.MAX_LEN, target_list)
valid_dataset = YicheVideosDataset(val_df, tokenizer, Config.MAX_LEN, target_list)
test_dataset = YicheVideosDataset(test_df, tokenizer, Config.MAX_LEN, target_list)
target_list = ['谍照', '实车曝光', '配置曝光', '申报图', '预热', '新车上市', '预售', '发布亮相', '新车官图',
               '新车报价', '新车到店', '新车解析']
'''
载入loader，也就是可以对dataset执行batch等操作,但是我们相对train dataset进行增强，这种直接切分的方式并不方便
训练集的数据加载器（train_data_loader）使用 shuffle=True 是正确的，这有助于在每个训练周期中随机打乱数据，从而防止模型过拟合。
验证集的数据加载器（val_data_loader）通常设置为 shuffle=False，因为验证过程中不需要随机打乱数据。验证集主要用于评估模型的表现，因此其数据顺序不会影响性能评估结果
'''
train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=Config.TRAIN_BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=0
                                                )

'''
这个为啥不是shuffle
'''
val_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                              batch_size=Config.VALID_BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=0
                                              )

'''
测试集
'''
test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=Config.TEST_BATCH_SIZE,
                                               shuffle=False, num_workers=0
                                               )



'''
开始初始化这个模型
'''
model = YicheBERT_Class()
# model = YicheBERT_TEXTCNN_Class()
model.to(device)

'''
优化器
'''
optimizer = torch.optim.Adam(params = model.parameters(), lr=Config.LEARNING_RATE)

'''
模型加载
'''
def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_loss_min = checkpoint['valid_loss_min']
    return model, optimizer
    # return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    f_path = checkpoint_path
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)


'''
损失函数 BCEWITHLOGISTSLOSS - it is a cross entropy with sigmoid
'''
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def evaluate(model, data_loader):
    threshold = 0.5
    model.eval()
    total_loss = 0
    test_targets = []
    test_outputs = []

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader, 0):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)
            loss = loss_fn(outputs, targets)
            total_loss += ((1 / (batch_idx + 1)) * (loss.item() - total_loss))

            test_targets.append(targets.cpu().detach().numpy())
            test_outputs.append(torch.sigmoid(outputs).cpu().detach().numpy())

    test_targets = np.concatenate(test_targets)
    test_outputs = np.concatenate(test_outputs) > threshold
    precision, recall, f1, _ = precision_recall_fscore_support(test_targets, test_outputs, average=None,
                                                               zero_division=0)

    with open('train_news_video_p0_wwm_bert.log', 'a', encoding='utf-8') as log_file:
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            label_name = target_list[i]
            log_info = f'Label {label_name} - Precision: {p:.4f}, Recall: {r:.4f}, F1-Score: {f:.4f}\n'
            print(log_info)
            log_file.write(log_info)


    return total_loss / len(data_loader)

'''
把model扔到这里面进行训练
'''
def train_model(n_epochs, training_loader, validation_loader, model,
                   optimizer, checkpoint_path, best_model_path):
    # 初始化为无穷大
    valid_loss_min = np.Inf
    threshold = 0.5
    '''
    每隔500次batch就测试一下
    '''
    eval_every_n_batches = 100

    for epoch in range(1, n_epochs + 1):
        train_loss = 0
        train_targets = []
        train_outputs = []

        model.train()
        print('############# Epoch {}: Training Start #############'.format(epoch))
        for batch_idx, data in enumerate(training_loader):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
            train_targets.append(targets.cpu().detach().numpy())
            train_outputs.append(torch.sigmoid(outputs).cpu().detach().numpy())
            '''
            训练阶段不需要得到训练数据的P,R和F1值，因为意义不大
            '''
            print(f'Epoch: {epoch} Training Batch {batch_idx + 1} - Loss: {loss.item():.6f}')

        print('############# Epoch {}: Training End #############'.format(epoch))

        model.eval()
        valid_loss = 0
        val_targets = []
        val_outputs = []
        print('############# Epoch {}: Validation Start #############'.format(epoch))
        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader, 0):
                ids = data['input_ids'].to(device, dtype=torch.long)
                mask = data['attention_mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)

                outputs = model(ids, mask, token_type_ids)
                loss = loss_fn(outputs, targets)
                valid_loss += ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))

                val_targets.append(targets.cpu().detach().numpy())
                val_outputs.append(torch.sigmoid(outputs).cpu().detach().numpy())

                if (batch_idx + 1) % eval_every_n_batches == 0:
                    batch_targets = np.concatenate(val_targets)
                    batch_outputs = np.concatenate(val_outputs) > threshold
                    precision, recall, f1, _ = precision_recall_fscore_support(batch_targets, batch_outputs, average=None, zero_division=0)
                    print(f'Validation Batch {batch_idx + 1} - Loss: {loss.item():.6f}')

                    with open('train_news_video_p0_wwm_bert.log', 'a',encoding='utf-8') as log_file:
                        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
                            label_name = target_list[i]
                            log_info = f'Label {label_name} - Precision: {p:.4f}, Recall: {r:.4f}, F1-Score: {f:.4f}\n'
                            print(log_info)
                            log_file.write(log_info)

                    val_targets = []
                    val_outputs = []

        train_loss /= len(training_loader)
        valid_loss /= len(validation_loader)
        print(f'Epoch: {epoch} \tAverage Training Loss: {train_loss:.6f} \tAverage Validation Loss: {valid_loss:.6f}')

        # Checkpointing and saving best model
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        '''
        这个就是在调整训练的模型，也就是说，验证集不会改变模型的参数，但是当验证集发现训练的模型在val数据集上的loss无法继续优化后就中止继续训练，
        或者只保存最好的loss结果模型
        '''
        if valid_loss <= valid_loss_min:
            print(f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model ...')
            if not os.path.exists(checkpoint_path):
                os.mkdir(checkpoint_path)

            checkpoint_filename = os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_filename)
            best_model_filename = os.path.join(best_model_path, 'best_model.pt')
            if valid_loss == valid_loss_min:
                torch.save(model.state_dict(), best_model_filename)
            valid_loss_min = valid_loss

        print('############# Epoch {} Done #############\n'.format(epoch))

    return model


def convert_to_labels(label_list, label_indices):
    return [label for label, index in zip(label_list, label_indices) if index == 1]


def predict(model, data_loader, label_list):
    import csv
    threshold = 0.5
    model.eval()
    all_records = []

    with torch.no_grad():
        for data in data_loader:
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

    # 写入 CSV 文件
    with open('p0_news_video_bert_wwm_predict.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Id','Content', 'True Labels', 'Predicted Labels'])

        for record in all_records:
            content_id,content, true_labels_str, predicted_labels_str = record
            writer.writerow([content_id,content, true_labels_str, predicted_labels_str])



'''
bert-based-chinese: 下载模型
https://modelscope.cn/models/tiansz/bert-base-chinese/summary
nohup CUDA_VISIBLE_DEVICES=4 python train_video_news_bert_p0.py > train_video_news.log 2>&1 &
 
'''
if __name__ == '__main__':
    '''
    设置模型保存地址
    '''
    ckpt_path = os.path.join(os.getcwd(),'save_news_video_wwmlarge_models_for_p0','curr_ckpt')
    best_model_path = os.path.join(os.getcwd(),'save_news_video_wwmlarge_models_for_p0','best_model.pt')

    trained_model = train_model(Config.EPOCHS, train_data_loader, val_data_loader, model, optimizer, ckpt_path, best_model_path)
    print('############# Test Start #############')
    '''
    用测试集进行测试
    '''
    test_loss = evaluate(trained_model, test_data_loader)
    print(f'Test Loss: {test_loss:.6f}')
    print('############# Test End #############')

    '''
    产生输出用测试集
    '''
    checkpoint_model_path = os.path.join(os.getcwd(), 'save_news_video_wwmlarge_models_for_p0', 'curr_ckpt', 'checkpoint_epoch_8.pt')
    checkpoint_model, optimizer = load_ckp(checkpoint_model_path, model, optimizer)
    p0_label_list = ['谍照', '实车曝光', '配置曝光', '申报图', '预热', '新车上市', '预售', '发布亮相', '新车官图',
                  '新车报价', '新车到店', '新车解析']

    # p1_label_list =   ['车系品牌解读', '单车导购', '对比导购', '多车导购', '购车手册', '买车技巧', '评测导购',
    #          '汽车分享', '试驾', '探店报价', '无解说车辆展示', '营销导购']

    predict(model, test_data_loader, p0_label_list)
















