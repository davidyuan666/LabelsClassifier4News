# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta

# 全局配置
MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

def build_vocab(file_path, tokenizer, max_size, min_freq):
    """
    构建词表
    Args:
        file_path: 训练数据文件路径
        tokenizer: 分词器函数
        max_size: 词表最大大小
        min_freq: 最小词频
    Returns:
        vocab_dic: 词表字典 {word: index}
    """
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f, desc='Building Vocabulary'):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], 
                          key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic

def build_dataset(config, use_word):
    """
    构建训练数据集
    Args:
        config: 配置对象
        use_word: 是否使用词级别分词
    Returns:
        vocab: 词表
        train: 训练数据
    """
    # 选择分词方式
    tokenizer = lambda x: x.split(' ') if use_word else [y for y in x]
    
    # 加载或构建词表
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, 
                          max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):
        """加载数据集"""
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f, desc='Loading Dataset'):
                lin = line.strip()
                if not lin:
                    continue
                try:
                    content, label = lin.split('\t')
                except ValueError:
                    continue
                
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                
                # 处理序列长度
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                
                # 将词转换为索引
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    return vocab, train

class DatasetIterater(object):
    """
    数据集迭代器
    """
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        """转换为tensor"""
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches + 1 if self.residue else self.n_batches

def build_iterator(dataset, config):
    """构建数据迭代器"""
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

if __name__ == "__main__":
    """提取预训练词向量"""
    # 配置路径
    train_dir = "data/train.txt"
    vocab_dir = "data/vocab.pkl"
    pretrain_dir = "data/sgns.sogou.char"  # 预训练的词向量
    emb_dim = 300
    filename_trimmed_dir = "data/embedding_bitautoNews"  # 保存的词向量文件
    
    # 加载或构建词表
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, 
                               max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    # 加载预训练词向量
    embeddings = np.random.rand(len(word_to_id), emb_dim)
    print("Loading pretrained embeddings...")
    with open(pretrain_dir, "r", encoding='UTF-8') as f:
        for i, line in enumerate(tqdm(f.readlines())):
            lin = line.strip().split(" ")
            if lin[0] in word_to_id:
                idx = word_to_id[lin[0]]
                emb = [float(x) for x in lin[1:301]]
                embeddings[idx] = np.asarray(emb, dtype='float32')
    
    # 保存处理后的词向量
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
    print(f"Saved processed embeddings to {filename_trimmed_dir}.npz")