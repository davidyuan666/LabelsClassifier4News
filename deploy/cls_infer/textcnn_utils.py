# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta


MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

def build_dataset_from_api(inputs, config, use_word):
    if use_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level

    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        raise FileNotFoundError("Vocab file not found.")

    def load_dataset(inputs, pad_size=32):
        contents = []
        for input in inputs:
            words_line = []
            token = tokenizer(input)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))  # 使用配置中的PAD填充
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id，使用配置中的UNK处理未知词
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line,input, 0, seq_len))  # 默认标签为0，可以根据需要调整

        return contents

    contents = load_dataset(inputs, config.pad_size)
    return vocab, contents
