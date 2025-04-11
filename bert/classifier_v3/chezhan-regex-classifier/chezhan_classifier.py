# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/4 15:41
@Auth ： David Yuan
@File ：chezhan_classifier.py
@Institute ：BitAuto
"""
import os
import pandas as pd
import re
'''            
使用正则进行过滤p0中的车展
车展活动	国内车展	广州车展
                北京车展
                成都车展
                上海车展
                粤港澳车展
                重庆车展
                天津车展
        国外车展	洛杉矶车展
                巴黎车展
                北美车展
                东京车展
                慕尼黑车展
                纽约车展
                日内瓦车展
                CES展
                法兰克福车展
	    云车展	
'''
import re

chezhan_tag_name_map = {
    1000162: "车展活动",
    1000278: "国内车展",
    1000281: "广州车展",
    1000282: "北京车展",
    1000285: "粤港澳车展",
    1000279: "国外车展",
    1000288: "洛杉矶车展",
    1000290: "北美车展",
    1000294: "日内瓦车展",
}


# 三级标签的具体车展名称
tag_3_car_show_names = [
    "广州车展", "北京车展", "成都车展", "上海车展", "粤港澳车展", "重庆车展", "天津车展",
    "洛杉矶车展", "巴黎车展", "北美车展", "东京车展", "慕尼黑车展", "纽约车展", "日内瓦车展", "CES展", "法兰克福车展", "云车展"
]

# 二级标签和对应的正则表达式
tag_2_car_show_names = {
    "国内车展",
    "国外车展",
    "云车展"
}

# 匹配并分类
def classify_car_show(text):
    matched_labels = []

    # 匹配三级标签
    for show_name in tag_3_car_show_names:
        if re.search(show_name, text):
            matched_labels.append(show_name)

    # 匹配二级标签
    for show_name in tag_2_car_show_names:
        if re.search(show_name, text):
            matched_labels.append(show_name)

    # 匹配一级标签 - 车展活动
    if "车展活动" in text:
        matched_labels.append("车展活动")

    # 返回匹配到的所有标签，用逗号分隔，如果没有匹配到，返回"未知"
    return ', '.join(matched_labels) if matched_labels else "未知"


'''
按照标签分别输出到不同的sheet，这样方便业务进行查看
pip install pandas openpyxl
'''
import pandas as pd
from collections import defaultdict
def normalize_label(label):
    """标准化标签：去除空格、转为小写、去除特殊字符"""
    label = label.strip().lower()
    label = re.sub(r'\W+', '', label)  # 去除非字母数字字符
    return label

def output_in_labels():
    chezhan_file = os.path.join(os.getcwd(), 'other_data','车展_data_with_predictions_v2.csv')
    # 加载CSV文件
    df = pd.read_csv(chezhan_file, sep='\t', encoding='UTF-8')

    # 创建一个pandas的Excel写入器
    with pd.ExcelWriter(os.path.join(os.getcwd(),'other_data','车展output_v4_withcontentid.xlsx'), engine='openpyxl') as writer:
        # 获取所有唯一的预测标签（考虑到可能的组合）
        all_predicted_labels = set()
        for item in df['predict'].dropna().unique():
            labels = [normalize_label(label) for label in item.split(',')]
            all_predicted_labels.update(labels)

        # 对每个唯一标签处理
        for label in all_predicted_labels:
            # 筛选包含特定标签的数据
            label_df = df[df['predict'].apply(lambda x: normalize_label(label) in [normalize_label(l) for l in str(x).split(',')])]

            # 将数据写入工作表，工作表名称为标签名
            label_df.to_excel(writer, sheet_name=label[:31], index=False)  # Excel工作表名长度限制为31


def output_in_labels_v2():
    chezhan_file = os.path.join(os.getcwd(), 'other_data', '车展_data_with_predictions_v2.csv')
    # 加载CSV文件
    df = pd.read_csv(chezhan_file, sep='\t', encoding='UTF-8')

    # 创建一个pandas的Excel写入器
    with pd.ExcelWriter(os.path.join(os.getcwd(), 'other_data', '车展output_v4_withcontentid.xlsx'),
                        engine='openpyxl') as writer:
        # 获取所有唯一的预测标签
        all_predicted_labels = set()
        for item in df['predict'].dropna().unique():
            labels = [normalize_label(label) for label in item.split(',')]
            all_predicted_labels.update(labels)

        # 对每个唯一标签处理
        for label in all_predicted_labels:
            # 筛选包含特定标签的数据
            label_df = df[df['predict'].apply(
                lambda x: normalize_label(label) in [normalize_label(l) for l in str(x).split(',')])]

            # 确保在写入Excel时包括 contentid 列
            label_df.to_excel(writer, sheet_name=label[:31], index=False,
                              columns=['contentid', 'text', 'label', 'predict'])

    print('Excel file with contentid has been generated.')




import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def normalize_label(label):
    # 实现标签的标准化
    return label.lower().strip()

def find_closest_contentid(text, df_1, vectorizer):
    # 将输入文本与df_1中的文本进行比较，返回最相似的contentid
    tfidf_matrix = vectorizer.transform(df_1['summary'].tolist() + [text])  # 使用正确的列名
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    closest_index = cosine_similarities.argsort()[-1]
    return df_1.iloc[closest_index]['contentid']


import csv
from tqdm import tqdm
def add_csv_title():
    input_csv_filepath = os.path.join(os.getcwd(), 'other_data', '1229_newsvideo_fewshot.csv')
    output_csv_filepath = os.path.join(os.getcwd(), 'other_data', '1229_newsvideo_with_titles.csv')

    with open(input_csv_filepath, 'r', encoding='utf-8') as infile, \
         open(output_csv_filepath, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')

        writer.writerow(["contentid", "title", "summary", "labels", "dtype", "recommend_grade"])

        for row in tqdm(reader, desc="Processing 1229_newsvideo"):
            if len(row) != 6:
                print(f"Skipping row with unexpected number of columns: {row}")
                continue
            writer.writerow(row)

        print('New CSV with titles has been generated.')



'''
把包含车展的条目挑选出来
'''


def filter_chezhan():
    chezhan_tag_name_map = {
        1000162: "车展活动",
        1000278: "国内车展",
        1000281: "广州车展",
        1000282: "北京车展",
        1000285: "粤港澳车展",
        1000279: "国外车展",
        1000288: "洛杉矶车展",
        1000290: "北美车展",
        1000294: "日内瓦车展",
    }
    original_file = os.path.join(os.getcwd(), 'other_data', '1229_newsvideo_with_titles.csv')
    chezhan_filter_file = os.path.join(os.getcwd(), 'other_data', '车展_with_contentid.csv')

    df_1 = pd.read_csv(original_file, sep='\t', encoding='UTF-8')
    filtered_data = []

    for key, value in chezhan_tag_name_map.items():
        matched_rows = df_1[df_1['labels'].astype(str).str.contains(str(key))]
        for _, row in matched_rows.iterrows():
            if pd.notna(row['summary']) and row['summary'].strip():  # 检查 summary 是否非空
                filtered_data.append({'contentid': row['contentid'], 'label': value, 'text': row['summary']})

    df_2 = pd.DataFrame(filtered_data)
    df_2.to_csv(chezhan_filter_file, sep='\t', index=False, encoding='UTF-8')


if __name__ == '__main__':
    # filter_chezhan()
    # Define the path to the CSV file
    # chezhan_file = os.path.join(os.getcwd(), 'other_data', '车展_with_contentid.csv')
    # output_file = os.path.join(os.getcwd(), 'other_data', '车展_data_with_predictions_v2.csv')
    #
    # try:
    #     df = pd.read_csv(chezhan_file, sep='\t', encoding='UTF-8')
    # except pd.errors.ParserError as e:
    #     print(f"Error reading CSV file: {e}")
    #     exit(1)
    #
    # # Ensure the DataFrame contains necessary columns
    # for column in ['text', 'label', 'contentid']:
    #     if column not in df.columns:
    #         raise ValueError(f"CSV file must contain a '{column}' column.")
    #
    # # Apply classify_car_show function
    # df['predict'] = df['text'].apply(classify_car_show)
    #
    # # Select and save the specific columns
    # df[['text', 'label', 'predict', 'contentid']].to_csv(output_file, sep='\t', encoding='utf-8', index=False)
    #
    # print(f"Updated CSV with predictions saved as: {output_file}")

    # add_csv_title()
    output_in_labels_v2()