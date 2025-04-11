# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/23 20:26
@Auth ： David Yuan
@File ：dataset_loader.py
@Institute ：BitAuto
"""

import torch

'''
数据集定义对象yiche
'''
class YicheNewsDataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, max_len,target_list):
        self.tokenizer = tokenizer
        self.df = df
        if 'content_id' in df.columns:
            self.content_id = df['content_id']
        else:
            self.content_id = [''] * len(df)
        self.content = df['content']
        self.targets = self.df[target_list].values
        self.max_len = max_len

    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        content = str(self.content[index])
        content_id = str(self.content_id[index])
        '''
        把文本按照空格进行分词，适用于英文，中文需要其他的分词器和词编码方式
        '''
        content = " ".join(content.split())
        '''
        使用分词器
        '''
        inputs = self.tokenizer.encode_plus(
            content,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'content_ids':content_id,
            'contents': content,
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.FloatTensor(self.targets[index])
        }



'''
Yiche的视频相关的标签数据
'''
class YicheVideosDataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, max_len,target_list):
        self.tokenizer = tokenizer
        self.df = df
        if 'videoid' in df.columns:
            self.video_id = df['videoid']
        else:
            self.video_id = [''] * len(df)
        self.content = df['内容']
        self.targets = self.df[target_list].values
        self.max_len = max_len

    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        content = str(self.content[index])
        video_id = str(self.video_id[index])
        '''
        把文本按照空格进行分词，适用于英文，中文需要其他的分词器和词编码方式
        '''
        content = " ".join(content.split())
        '''
        使用分词器
        '''
        inputs = self.tokenizer.encode_plus(
            content,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'content_ids':video_id,
            'contents': content,
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.FloatTensor(self.targets[index])
        }