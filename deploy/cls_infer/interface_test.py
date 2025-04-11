# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/7 14:37
@Auth ： David Yuan
@File ：interface_test.py
@Institute ：BitAuto
"""
import os.path
import time

import requests
import json


# 读取本地文件
# 读取本地文件
def read_file(filename):
    contents_tags = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                contents_tags.append((parts[0], parts[1]))  # 假设文件是content\ttag这样的格式
    return contents_tags


# 发送请求到接口
# 分批发送请求
def send_batch_request(url, headers, batch, batch_id):
    contents, tags = zip(*batch)  # 解压内容和标签
    data = {
        "contents": list(contents),
        "request_id": str(batch_id),  # 使用批次ID作为request_id
        "content_ids": [str(i + 100001) for i in range(len(batch))],  # 生成示例content_ids
        "source_type": "news"
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json(), tags


# 主发送请求函数，处理分批逻辑
def send_request(contents_tags):
    url = "http://10.20.64.3:8009/tags/check"
    headers = {"Content-Type": "application/json"}

    batch_size = 100  # Maximum size of each batch
    responses = []

    for i in range(0, len(contents_tags), batch_size):
        start_time = time.time()
        batch = contents_tags[i:i + batch_size]
        batch_id = i // batch_size
        # if batch_id > 100:
        #     continue
        response, tags = send_batch_request(url, headers, batch, batch_id)

        responses.append((response, tags))
        end_time = time.time()
        print(f"Batch {batch_id} cost time: {end_time - start_time}s")

    return responses


import csv
def save_results_to_csv(all_responses):
    with open('results_comparison.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['Content ID','Content', 'Original Tag', 'Prediction p0',  'Prediction p1', 'Prediction p2'])

        for response, original_tags in all_responses:
            content_ids = response.get("content_ids", [])
            result = response.get("result", [])

            p0_results = result[0]['p0']
            p1_results = result[1]['p1']
            p2_results = result[2]['p2']

            for i, content_id in enumerate(content_ids):
                original_tag = original_tags[i]
                content = p0_results[i]['content']
                pred_p0 = p0_results[i]['prediction'] if i < len(p0_results) else 'N/A'
                pred_p1 = p1_results[i]['prediction'] if i < len(p1_results) else 'N/A'
                pred_p2 = p2_results[i]['prediction'] if i < len(p2_results) else 'N/A'
                if original_tag == '1':
                    writer.writerow([content_id,content, original_tag, pred_p0, pred_p1, pred_p2])


# 主函数
if __name__ == "__main__":
    filename = "test.txt"
    contents_tags = read_file(filename)
    all_responses = send_request(contents_tags)
    save_results_to_csv(all_responses)