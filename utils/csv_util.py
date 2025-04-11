# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/9 19:30
@Auth ： David Yuan
@File ：process.py
@Institute ：BitAuto
"""
import os.path

'''
把csv文本转为txt格式用来进行二分类评估
csv包含text,predicted_labels,real_labels 这些column，我们需要转化为txt格式其中每一行是文本，用\t分割成0或者1，这里面默认初始为-1,
csv 名字叫output_filter.csv 输出为new_dev_配置曝光.txt
'''
import pandas as pd

def convert_csv_to_txt_for_labels(csv_filename, labels):
    df = pd.read_csv(csv_filename,sep='\t',encoding='UTF-8')

    for label in labels:
        txt_filename =os.path.join(os.getcwd(),'yiche-news-summary','data',f"p1_dev.txt")
        with open(txt_filename, 'w', encoding='utf-8') as txt_file:
            if 'text' not in df.columns:
                raise ValueError("CSV file must contain a 'text' column.")
            for _, row in df.iterrows():
                text = row['text']
                # 默认为0是负样本
                label = 0
                txt_file.write(f"{text}\t{label}\n")


# Re-importing necessary libraries after the reset
import pandas as pd
import os


def merge_csv():
    csv_file1 = os.path.join(os.getcwd(), 'dataset', '20230101_news_and_yichehao_news.csv')
    csv_file2 = os.path.join(os.getcwd(), 'dataset', '20230101_yichehao_video.csv')
    output_csv_file = os.path.join(os.getcwd(), 'dataset', '20230101_combined_news_and_video.csv')

    df1 = pd.read_csv(csv_file1, sep='\t', header=None, on_bad_lines='skip')
    df2 = pd.read_csv(csv_file2, sep='\t', header=None, on_bad_lines='skip')

    combined_df = pd.concat([df1, df2])

    combined_df.drop_duplicates(subset=[0], inplace=True)

    combined_df['merged_content'] = combined_df[1] + " " + combined_df[2]

    final_df = combined_df[[0, 'merged_content', 3]].rename(columns={0: 'id', 'merged_content': 'content', 3: 'tags'})

    final_df.to_csv(output_csv_file, index=False, sep='\t', header=False)


def filter_tag_ids():
    csv_file_path = os.path.join(os.getcwd(), 'dataset', '20230601_tag.csv')

    df = pd.read_csv(csv_file_path, sep='\t', header=None,
                     names=['leaf_tag_id', 'tag_id_hierarchy', 'tag_name_hierarchy'])

    # 定义目标标签列表
    p0_target_list = ['谍照', '实车曝光', '配置曝光', '申报图', '预热', '新车上市', '预售', '发布亮相', '新车官图',
                      '新车报价', '新车到店', '新车解析']
    p1_target_list = ['车系品牌解读', '单车导购', '对比导购', '多车导购', '购车手册', '买车技巧', '评测导购',
                      '汽车分享', '试驾', '探店报价', '无解说车辆展示', '营销导购']
    p2_target_list = ['交通事故', '自燃', '维权事件', '车辆减配', '故障投诉', '车辆召回', '产能不足', '车辆首撞',
                      '商家吐槽', '爱车吐槽', '交通政策', '补贴政策', '汽车油价', '二手车限迁法规', '价格行情',
                      '花边新闻', '销量新闻', '新闻聚合', '人物观点', '行业评论', '汽车出口', '新能源新闻', '论坛峰会']


    # 分别过滤并保存p0, p1, p2目标标签号码
    filter_and_save_tag_ids_with_names(p0_target_list, df, 'dataset/p0_tag_ids.txt')
    filter_and_save_tag_ids_with_names(p1_target_list, df, 'dataset/p1_tag_ids.txt')
    filter_and_save_tag_ids_with_names(p2_target_list, df, 'p2_tag_ids.txt')


def filter_and_save_tag_ids_with_names(target_list, df, output_filename):
    tag_name_id_map = {}
    for _, row in df.iterrows():
        tag_names = row['tag_name_hierarchy'].split(',')
        for target_tag in target_list:
            if target_tag in tag_names:
                leaf_tag_id = row['leaf_tag_id']
                tag_name_id_map[target_tag] = leaf_tag_id

    with open(output_filename, 'w') as f:
        for tag_name, tag_id in tag_name_id_map.items():
            f.write(f"{tag_name}: {tag_id}\n")

    print(f"Saved {len(tag_name_id_map)} tag names and IDs to {output_filename}")


def generate_txt_for_binary_classification_p0():
    input_csv_path = os.path.join(os.getcwd(), 'dataset', '20230101_combined_news_and_video.csv')
    output_txt_path = os.path.join(os.getcwd(), 'dataset', 'p0.txt')

    # 目标标签号码列表 for p0
    target_tag_ids_p0 = [
        "58",  # 配置曝光
        "1000130",  # 谍照
        "1000121",  # 实车曝光
        "1000120",  # 申报图
        "1000118",  # 预热
        "1000126",  # 预售
        "1000125",  # 新车上市
        "1000127",  # 发布亮相
        "1000539",  # 新车官图
        "1000540",  # 新车到店
        "1000541",  # 新车解析
        "1000516"  # 新车报价
    ]

    # 读取CSV文件
    df = pd.read_csv(input_csv_path, sep='\t', header=None, names=['id', 'content', 'tags'])

    # 打开输出文件
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # 分割标签字符串，得到标签号码列表
            tags = str(row['tags']).split(',')
            # 检查是否包含目标标签号码
            label = 1 if any(tag in target_tag_ids_p0 for tag in tags) else 0
            # 将内容和标签写入输出文件
            f.write(f"{row['content']}\t{label}\n")

    print(f"Binary classification data saved to {output_txt_path}")

def generate_txt_for_binary_classification_p1():
    input_csv_path = os.path.join(os.getcwd(), 'dataset', '20230101_combined_news_and_video.csv')
    output_txt_path = os.path.join(os.getcwd(), 'dataset', 'p1.txt')

    # 目标标签号码列表
    target_tag_ids_p1 = ['1000012', '1000316', '1000313', '1000298', '1000026', '1000019', '1000014', '1000013', '1000538',
                      '1000537', '1000536', '1000535']

    # 读取CSV文件
    df = pd.read_csv(input_csv_path, sep='\t', header=None, names=['id', 'content', 'tags'])

    # 打开输出文件
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # 分割标签字符串，得到标签号码列表
            tags = str(row['tags']).split(',')
            # 检查是否包含目标标签号码
            label = 1 if any(tag in target_tag_ids_p1 for tag in tags) else 0
            # 将内容和标签写入输出文件
            f.write(f"{row['content']}\t{label}\n")

    print(f"Binary classification data saved to {output_txt_path}")


def generate_txt_for_binary_classification_p2():
    input_csv_path = os.path.join(os.getcwd(), 'dataset', '20230101_combined_news_and_video.csv')
    output_txt_path = os.path.join(os.getcwd(), 'dataset', 'p2.txt')

    # 目标标签号码列表
    target_tag_ids_p2 = [
        '1000048', '1000528', '1000151', '1000171', '1000199', '1000469', '1000139', '555',
        '1000322', '1000271', '1000146', '1000144', '1000143', '1000141', '1000084', '611',
        '1000149', '338', '1000520', '1000545', '1000544', '1000543', '1000542'
    ]

    # 读取CSV文件
    df = pd.read_csv(input_csv_path, sep='\t', header=None, names=['id', 'content', 'tags'])

    # 打开输出文件
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # 分割标签字符串，得到标签号码列表s
            tags = str(row['tags']).split(',')
            # 检查是否包含目标标签号码
            label = 1 if any(tag in target_tag_ids_p2 for tag in tags) else 0
            # 将内容和标签写入输出文件
            f.write(f"{row['content']}\t{label}\n")

    print(f"Binary classification data saved to {output_txt_path}")


def calculate_label_ratio_and_split_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    label_count = {"1": 0, "0": 0}

    for line in lines:
        if 'nan' not in line.lower():
            try:
                # Split the line from the right, ensuring only two parts are returned
                content, label = line.rsplit('\t', 1)
                label = label.strip()
                if label in label_count:
                    label_count[label] += 1
            except ValueError:
                # Handle lines that don't conform to the expected format
                print(f"Skipping line due to formatting issues: {line}")

    total_labels = label_count["1"] + label_count["0"]
    ratio_1 = label_count["1"] / total_labels
    ratio_0 = label_count["0"] / total_labels

    # Splitting the dataset into train, dev, and test sets
    train_lines = lines[:int(len(lines) * 0.8)]
    dev_lines = lines[int(len(lines) * 0.8):int(len(lines) * 0.9)]
    test_lines = lines[int(len(lines) * 0.9):]

    # Save the splits to their respective files
    # Ensure to replace 'p0' with the appropriate prefix for p1 and p2 datasets
    with open(os.path.join(os.getcwd(),'train_dataset','p2_train.txt'), 'w', encoding='utf-8') as train_file:
        train_file.writelines([line for line in train_lines if 'nan' not in line])
    with open(os.path.join(os.getcwd(),'train_dataset','p2_dev.txt'), 'w', encoding='utf-8') as dev_file:
        dev_file.writelines([line for line in dev_lines if 'nan' not in line])
    with open(os.path.join(os.getcwd(),'train_dataset','p2_test.txt'), 'w', encoding='utf-8') as test_file:
        test_file.writelines([line for line in test_lines if 'nan' not in line])

    return ratio_1, ratio_0


'''
p0:
1: 0.22062302881331616
0: 0.7793769711866838
p1:
1: 0.2276117086080237
0: 0.7723882913919763
p2:
1: 0.11628315287094709
0: 0.883716847129053
'''
if __name__ == '__main__':
    # merge_csv()
    # filter_tag_ids()
    # generate_txt_for_binary_classification_p0()
    # generate_txt_for_binary_classification_p1()
    # generate_txt_for_binary_classification_p2()
    input_txt_path = os.path.join(os.getcwd(), 'dataset', 'p2.txt')
    ratio_1_p0, ratio_0_p0= calculate_label_ratio_and_split_data(input_txt_path)
    print(f'1: {ratio_1_p0}')
    print(f'0: {ratio_0_p0}')