# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/19 15:29
@Auth ： David Yuan
@File ：process.py
@Institute ：BitAuto
"""
import os
from sklearn.model_selection import train_test_split


def split_into_train_test_dev(yiche_origin_path, train_ratio=0.8, test_ratio=0.1, dev_ratio=0.1):
    assert train_ratio + test_ratio + dev_ratio == 1, "The sum of ratios must be 1."

    train_txt_path = os.path.join(os.getcwd(), 'datasets', 'train.txt')
    test_txt_path = os.path.join(os.getcwd(), 'datasets', 'test_bk.txt')
    val_txt_path = os.path.join(os.getcwd(), 'datasets', 'dev.txt')

    if not os.path.exists(yiche_origin_path):
        print(f"The file {yiche_origin_path} does not exist.")
        return

    with open(yiche_origin_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    train_temp, test_data = train_test_split(lines, test_size=(test_ratio + dev_ratio), random_state=42)
    test_data, val_data = train_test_split(test_data, test_size=dev_ratio / (test_ratio + dev_ratio), random_state=42)

    with open(train_txt_path, 'w', encoding='utf-8') as file:
        file.writelines(train_temp)
    with open(test_txt_path, 'w', encoding='utf-8') as file:
        file.writelines(test_data)
    with open(val_txt_path, 'w', encoding='utf-8') as file:
        file.writelines(val_data)

    print("Data has been split and saved successfully.")




if __name__ == '__main__':
    yiche_origin_path = os.path.join(os.getcwd(), 'datasets', 'yiche_news_pretrain.txt')
    split_into_train_test_dev(yiche_origin_path)