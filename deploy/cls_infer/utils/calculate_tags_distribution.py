# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/12 17:57
@Auth ： David Yuan
@File ：calculate_tags_distribution.py
@Institute ：BitAuto
"""
import pandas as pd
import csv
import os

def create_tag_dictionary(tag_csv_path):
    tag_df = pd.read_csv(tag_csv_path, header=None, sep='\t')
    tag_dict = {}
    for _, row in tag_df.iterrows():
        tag_ids = row[1].split(',')
        tag_names = row[2].split(',')
        for tag_id, tag_name in zip(tag_ids, tag_names):
            tag_dict[int(tag_id)] = tag_name
    return tag_dict



def count_tags(news_csv_path, tag_dict, p3_tags, p4_tags):
    news_df = pd.read_csv(news_csv_path, sep='\t', header=None)
    p3_counts = {tag: 0 for tag in p3_tags}
    p4_counts = {tag: 0 for tag in p4_tags}

    for _, row in news_df.iterrows():
        tags = [int(tag) for tag in row[3].split(',') if tag.isdigit()]
        for tag in tags:
            tag_name = tag_dict.get(tag)
            if tag_name in p3_tags:
                p3_counts[tag_name] += 1
            if tag_name in p4_tags:
                p4_counts[tag_name] += 1

    return p3_counts, p4_counts



def reverse_dict(input_dict):
    reversed_dict = {value: key for key, value in input_dict.items()}
    return reversed_dict


def tags_to_dict(tags_list, tag_dict):
    tags_with_ids = {}
    for tag in tags_list:
        tag_id = tag_dict.get(tag)  # 直接使用 get 方法来查找标签对应的编号
        if tag_id:
            tags_with_ids[tag] = tag_id
    return tags_with_ids



def analyze_tags(df, tag_dict):
    # 假设第四列包含标签ID
    tag_column_index = 3  # 因为列索引从0开始，所以第四列索引是3
    tag_counts = {}

    # 遍历DataFrame中的每一行
    for index, row in df.iterrows():
        tag_ids = str(row[tag_column_index]).split(',')
        for tag_id in tag_ids:
            if tag_id.isdigit() and int(tag_id) in tag_dict:  # 检查tag_id是否为数字且存在于tag_dict中
                if tag_id in tag_counts:
                    tag_counts[tag_id] += 1
                else:
                    tag_counts[tag_id] = 1

    return tag_counts


def display_tag_names(tag_counts, tag_dict):
    tag_name_counts = {}
    for tag_id_str, count in tag_counts.items():
        # 将字符串ID转换为整数，因为字典中的键是整数
        tag_id = int(tag_id_str)
        if tag_id in tag_dict:
            tag_name = tag_dict[tag_id]
            if tag_name in tag_name_counts:
                tag_name_counts[tag_name] += count
            else:
                tag_name_counts[tag_name] = count
        else:
            # 如果标签ID在字典中不存在，将其分类到"未知标签"
            if "未知标签" in tag_name_counts:
                tag_name_counts["未知标签"] += count
            else:
                tag_name_counts["未知标签"] = count

    return tag_name_counts


def read_and_process_csv(input_csv_path):
    try:
        # 使用 on_bad_lines='skip' 来跳过列数不符合预期的行
        df = pd.read_csv(input_csv_path, sep='\t', header=None, on_bad_lines='skip')
        print("CSV文件已成功载入，格式错误的行已被跳过。")
        return df
    except Exception as e:
        print(f"读取CSV文件时出错：{e}")


def process_p3_p4_distribution():
    input_csv_path = os.path.join(os.getcwd(), 'doc', '20231001_news_and_yichehao_news.csv')
    tag_csv_path = os.path.join(os.getcwd(), 'doc', '20230601_tag.csv')
    tag_dict = create_tag_dictionary(tag_csv_path)
    reversed_tag_dict = reverse_dict(tag_dict)  # 反转字典，使标签名称成为键

    print(reversed_tag_dict)


    p3_1_tags = ['技术与评测']
    p3_2_tags = ['评测']
    p3_3_tags = ['静态评测', '动态评测', '安全测试', '能耗测试', '音响测试', '噪音测试', '性能测试', '长测', '对比评测', '横评']
    p3_4_tags = ['拆解评测', '构造评测', '碰撞测试']

    p4_1_tags = ['技术与测评', '汽车周边']
    p4_2_tags = ['汽车技术', '汽车活动', '驾驶记录', '美图分享', '汽车历史', '海外车事']
    p4_3_tags = ['三大件技术', '安全技术', '生产与制造', '智能驾驶', '新能源技术', '车友聚会', '赛事活动', '汽车美图', '美女车模', '经典车历史', '老车修复', '品牌成长史', '情怀老车', '海外新车', '海外评测']
    p4_4_tags = ['结构与材料', '被动安全', 'F1方程式', '老爷车']

    p3_1_dict = tags_to_dict(p3_1_tags, reversed_tag_dict)
    p3_2_dict = tags_to_dict(p3_2_tags, reversed_tag_dict)
    p3_3_dict = tags_to_dict(p3_3_tags, reversed_tag_dict)
    p3_4_dict = tags_to_dict(p3_4_tags, reversed_tag_dict)
    p4_1_dict = tags_to_dict(p4_1_tags, reversed_tag_dict)
    p4_2_dict = tags_to_dict(p4_2_tags, reversed_tag_dict)
    p4_3_dict = tags_to_dict(p4_3_tags, reversed_tag_dict)
    p4_4_dict = tags_to_dict(p4_4_tags, reversed_tag_dict)
    print('=====================================')
    print("P3_1 Tags with IDs:", p3_1_dict)
    print("P3_2 Tags with IDs:", p3_2_dict)
    print("P3_3 Tags with IDs:", p3_3_dict)
    print("P3_4 Tags with IDs:", p3_4_dict)
    print("P4_1 Tags with IDs:", p4_1_dict)
    print("P4_2 Tags with IDs:", p4_2_dict)
    print("P4_3 Tags with IDs:", p4_3_dict)
    print("P4_4 Tags with IDs:", p4_4_dict)


    df = read_and_process_csv(input_csv_path)
    # 调用函数
    if df is not None:
        tag_counts = analyze_tags(df,tag_dict)
        print("标签计数结果：")
        print(tag_counts)
        print('=================================')
        translated_tag_counts = display_tag_names(tag_counts, tag_dict)
        for tag_name, count in translated_tag_counts.items():
            print(f"{tag_name}: {count}")

    print('===========================')
    read_csv_and_calculate_tags(input_csv_path, tag_dict)

    print('===========================')
    # 将标签名称转换为标签ID集合
    def tags_to_ids(tags_list, tag_dict):
        ids = set()
        for tag in tags_list:
            ids.update(tag_dict[tag])
        return ids

    # 定义各分类的标签ID集
    categories = {
        'p3_1': tags_to_ids(['技术与评测'], tag_dict),
        'p3_2': tags_to_ids(['评测'], tag_dict),
        'p3_3': tags_to_ids(
            ['静态评测', '动态评测', '安全测试', '能耗测试', '音响测试', '噪音测试', '性能测试', '长测', '对比评测',
             '横评'], tag_dict),
        'p3_4': tags_to_ids(['拆解评测', '构造评测', '碰撞测试'], tag_dict),
        'p4_1': tags_to_ids(['技术与评测', '汽车周边'], tag_dict),
        'p4_2': tags_to_ids(['汽车技术', '汽车活动', '驾驶记录', '美图分享', '汽车历史', '海外车事'], tag_dict),
        'p4_3': tags_to_ids(
            ['三大件技术', '安全技术', '生产与制造', '智能驾驶', '新能源技术', '车友聚会', '赛事活动', '汽车美图',
             '美女车模', '经典车历史', '老车修复', '品牌成长史', '情怀老车', '海外新车', '海外评测'], tag_dict),
        'p4_4': tags_to_ids(['结构与材料', '被动安全', 'F1方程式', '老爷车'], tag_dict)
    }

    def read_and_process_csv_v2(input_csv_path, categories):
        df = pd.read_csv(input_csv_path, sep='\t', header=None, on_bad_lines='skip')
        tag_ids_column = df[3]

        category_counts = {key: 0 for key in categories}
        total_count = 0

        for tag_ids in tag_ids_column:
            tags = map(int, str(tag_ids).split(','))
            for tag in tags:
                for category, ids in categories.items():
                    if tag in ids:
                        category_counts[category] += 1
                        total_count += 1

        # 计算比例并显示
        for category, count in category_counts.items():
            percentage = (count / total_count) * 100 if total_count else 0
            print(f"{category}: {percentage:.2f}%")

    read_and_process_csv_v2(input_csv_path, categories)

def read_csv_and_calculate_tags(input_csv_path, tag_dict):
    try:
        # 读取CSV文件
        df = pd.read_csv(input_csv_path, sep='\t', header=None, on_bad_lines='skip')
        # 第四列包含标签ID，假设从0开始计数则为第3列
        tag_ids_column = df[3]

        # 初始化标签计数字典
        tag_counts = {}
        total_count = 0

        # 统计每个标签的出现次数
        for tag_ids in tag_ids_column:
            if pd.isna(tag_ids):
                continue
            tags = str(tag_ids).split(',')
            for tag in tags:
                if tag.isdigit():  # 确保是数字
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
                    total_count += 1

        # 计算每个标签的比例并打印
        print("标签占比：")
        for tag, count in tag_counts.items():
            percentage = (count / total_count) * 100
            tag_name = tag_dict.get(int(tag), "未知标签")
            print(f"{tag_name} ({tag}): {percentage:.2f}%")




    except Exception as e:
        print(f"读取或处理CSV文件时出错：{e}")








if __name__ == '__main__':
    process_p3_p4_distribution()