# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/12 16:49
@Auth ： David Yuan
@File ：test-api.py
@Institute ：BitAuto
"""

'''
常规库
'''
import requests
import re
import os
import pandas as pd
import pandas as pd
import csv
import time
import json

'''
导入配置文件
'''
from config import (
    SERVER_URL, DEFAULT_PORT, ENDPOINT,
    DEFAULT_PROMPT_VERSION, DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P, DEFAULT_MAX_TOKENS,TEST_COUNT
)


def start_test_inference_api():
    input_file_path = 'doc/yichenews_gpt4.csv'  # 确保文件路径和扩展名正确
    output_file_path = 'yichenews_gpt4_with_model.csv'

    summary_df = pd.read_csv(input_file_path)

    all_start_time = time.time()
    partial_start_time = time.time()  # 初始化部分时间记录器

    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ContentID', 'OriginalContent', 'gptAnswer', 'count', 'modelAnswer'])  # 修复了标题行

        for index, row in enumerate(summary_df.itertuples(), 1):
            content_id = row.ContentID
            original_content = row.OriginalContent
            gptAnswer = row.gptAnswer
            count = row.count
            model_answer = ''

            try:
                data = {"content": original_content}
                start_time = time.time()  # 开始请求前的时间
                response = requests.post(f'{SERVER_URL}:{DEFAULT_PORT}/infer', json=data)
                end_time = time.time()  # 请求完成的时间

                if response.status_code == 200:
                    model_answer = response.json().get('response', '')  # 优化了响应数据的获取
                    print(f"Processed {index}: {end_time - start_time} seconds")
                    print(f'====>\n'
                          f'{model_answer}')
                else:
                    model_answer = f"Error: Response code {response.status_code}"
            except Exception as e:
                model_answer = f"Exception occurred: {str(e)}"

            writer.writerow([content_id, original_content, gptAnswer, count, model_answer])

            if index % TEST_COUNT == 0:
                partial_end_time = time.time()
                print(f'Processed {TEST_COUNT} records in {partial_end_time - partial_start_time} seconds')
                partial_start_time = time.time()  # 重置部分时间记录器

    all_end_time = time.time()
    print(f'Total processing time: {all_end_time - all_start_time} seconds')



if __name__ == '__main__':
    start_test_inference_api()