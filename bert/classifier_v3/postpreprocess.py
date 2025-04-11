# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/25 17:15
@Auth ： David Yuan
@File ：postpreprocess.py
@Institute ：BitAuto
"""
import os
import pandas as pd

def filter_and_split_to_excel(input_csv, output_excel):
    target_list = ['谍照', '实车曝光', '配置曝光', '申报图', '预热', '新车上市', '预售', '发布亮相', '新车官图',
                   '新车报价', '新车到店', '新车解析']

    # 读取CSV文件
    df = pd.read_csv(input_csv)

    # 过滤掉预测标签或真实标签为null的行
    filtered_df = df.dropna(subset=['True Labels', 'Predicted Labels'])

    # 创建一个Excel写入器
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # 遍历目标标签列表
        for label in target_list:
            # 选择包含当前目标标签的行并去除contentid列重复的行
            label_df = filtered_df[filtered_df['Predicted Labels'].str.contains(label)].drop_duplicates(subset=['ID'])

            # 如果没有匹配的行或者处理后的行数为0，则继续下一个标签
            if label_df.empty:
                continue

            # 限制每个工作表的行数为200，如果超过，只取前200行
            label_df = label_df.head(200)

            # 缩短工作表名称，如果超过31字符
            sheet_name = label[:31]

            # 写入Excel的不同工作表
            label_df.to_excel(writer, sheet_name=sheet_name, index=False)


def filter_and_split_to_excel_v2(input_csv, output_excel):
    target_list = ['谍照', '实车曝光', '配置曝光', '申报图', '预热', '新车上市', '预售', '发布亮相', '新车官图',
                   '新车报价', '新车到店', '新车解析']

    # 读取CSV文件
    df = pd.read_csv(input_csv)

    # 过滤掉预测标签或真实标签为null的行
    filtered_df = df.dropna(subset=['True Labels', 'Predicted Labels'])

    # 创建一个Excel写入器
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # 遍历目标标签列表
        for label in target_list:
            # 初始化一个空的DataFrame用于收集每个dtype的数据
            collected_df = pd.DataFrame()

            # 对于新闻和视频类型分别处理
            for dtype in ['news', 'video']:
                # 选择当前dtype和包含当前目标标签的行
                dtype_df = filtered_df[
                    (filtered_df['dtype'] == dtype) & (filtered_df['Predicted Labels'].str.contains(label))]

                # 去除ID列重复的行
                dtype_df = dtype_df.drop_duplicates(subset=['ID'])

                # 如果数据超过100行，则只取前100行
                dtype_df = dtype_df.head(100)

                # 将筛选后的数据加入到收集用的DataFrame中
                collected_df = pd.concat([collected_df, dtype_df])

            # 如果没有匹配的行或者处理后的行数为0，则继续下一个标签
            if collected_df.empty:
                continue

            # 缩短工作表名称，如果超过31字符
            sheet_name = label[:31]

            # 写入Excel的不同工作表
            collected_df.to_excel(writer, sheet_name=sheet_name, index=False)



def filter_and_split_p1_to_excel(input_csv, output_excel):
    target_list = ['车系品牌解读', '单车导购', '对比导购', '多车导购', '购车手册', '买车技巧', '评测导购', '汽车分享',
                   '试驾', '探店报价', '无解说车辆展示', '营销导购']

    # 读取CSV文件
    df = pd.read_csv(input_csv)

    # 过滤掉预测标签或真实标签为null的行
    filtered_df = df.dropna(subset=['True Labels', 'Predicted Labels'])

    # 创建一个Excel写入器
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # 遍历目标标签列表
        for label in target_list:
            # 选择包含当前目标标签的行并去除contentid列重复的行
            label_df = filtered_df[filtered_df['Predicted Labels'].str.contains(label)].drop_duplicates(subset=['ContentID'])

            # 如果没有匹配的行或者处理后的行数为0，则继续下一个标签
            if label_df.empty:
                continue

            # 限制每个工作表的行数为200，如果超过，只取前200行
            label_df = label_df.head(200)

            # 缩短工作表名称，如果超过31字符
            sheet_name = label[:31]

            # 写入Excel的不同工作表
            label_df.to_excel(writer, sheet_name=sheet_name, index=False)


def filter_and_split_p2_to_excel(input_csv, output_excel):
    # target_list = ['交通政策','补贴政策','汽车油价','二手车限迁法规','价格行情','花边新闻','销量新闻','新闻聚合','人物观点','行业评论','汽车出口','新能源新闻','论坛峰会']
    target_list =  ['交通事故', '自燃','维权事件','车辆减配','故障投诉','车辆召回','产能不足','车辆首撞','商家吐槽','爱车吐槽']
    # 读取CSV文件
    df = pd.read_csv(input_csv)

    # 过滤掉预测标签或真实标签为null的行
    filtered_df = df.dropna(subset=['True Labels', 'Predicted Labels'])

    # 创建一个Excel写入器
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # 遍历目标标签列表
        for label in target_list:
            # 选择包含当前目标标签的行并去除contentid列重复的行
            label_df = filtered_df[filtered_df['Predicted Labels'].str.contains(label)].drop_duplicates(subset=['ContentID'])

            # 如果没有匹配的行或者处理后的行数为0，则继续下一个标签
            if label_df.empty:
                continue

            # 限制每个工作表的行数为200，如果超过，只取前200行
            label_df = label_df.head(200)

            # 缩短工作表名称，如果超过31字符
            sheet_name = label[:31]

            # 写入Excel的不同工作表
            label_df.to_excel(writer, sheet_name=sheet_name, index=False)


def merge_news_and_video_for_train():
    # 定义文件路径
    input_news_csv = os.path.join(os.getcwd(), 'data', 'bert_multilabel_cls_p0.csv')
    input_video_csv = os.path.join(os.getcwd(), 'data', '20231101_yiche_video_p0.csv')
    output_merge_csv = os.path.join(os.getcwd(), 'data', 'yiche_video_news_p0.csv')

    # 读取新闻数据并去掉最后一列以及新车资讯列
    news_data = pd.read_csv(input_news_csv)
    news_data = news_data.drop(columns=['其他', '新车资讯'])

    # 读取视频数据
    video_data = pd.read_csv(input_video_csv)

    # 将ContentID列重命名为videoid以匹配视频数据
    news_data.rename(columns={'ContentID': 'videoid'}, inplace=True)

    # 合并两个数据集
    merged_data = pd.concat([news_data, video_data], ignore_index=True)

    # 保存合并后的数据到CSV
    merged_data.to_csv(output_merge_csv, index=False)

    return merged_data.head()


def merge_news_and_video_for_prediction():
    # 定义输入文件和输出文件路径
    input_news_csv = os.path.join(os.getcwd(), 'data', 'P0_bert_wwmlarge_prediction.csv')
    input_video_csv = os.path.join(os.getcwd(), 'data', 'P0_bert_wwmlarge_video_prediction.csv')
    output_merge_csv = os.path.join(os.getcwd(), 'data', 'P0_bert_wwmlarge_all_prediction.csv')

    # 读取新闻数据，添加dtype列，并将ContentID重命名为ID
    news_df = pd.read_csv(input_news_csv)
    news_df.rename(columns={'ContentID': 'ID'}, inplace=True)
    news_df['dtype'] = 'news'

    # 读取视频数据，添加dtype列，并将VideoId重命名为ID
    video_df = pd.read_csv(input_video_csv)
    video_df.rename(columns={'VideoId': 'ID'}, inplace=True)
    video_df['dtype'] = 'video'

    # 合并两个DataFrame
    merged_df = pd.concat([news_df, video_df], ignore_index=True)

    # 保存到新的CSV文件
    merged_df.to_csv(output_merge_csv, index=False)




if __name__ == '__main__':
    # input_csv = os.path.join(os.getcwd(),'data','P0_bert_prediction_result_v2.csv')
    # output_excel = os.path.join(os.getcwd(),'data','P0_bert_prediction_v3.xlsx')

    # input_csv = os.path.join(os.getcwd(),'data','p2_政府车圈新闻_bert_wwmlarge_prediction.csv')
    # output_excel = os.path.join(os.getcwd(),'data','P2_政府车圈新闻_标签优化.xlsx')

    input_csv = os.path.join(os.getcwd(), 'data', 'p2_内容负向_bert_wwm_large_predict.csv')
    output_excel = os.path.join(os.getcwd(), 'data', 'P2_内容负向_标签优化.xlsx')

    # 执行转换
    # filter_and_split_to_excel(input_csv, output_excel)

    # filter_and_split_to_excel_v2(input_csv,output_excel)

    # filter_and_split_p1_to_excel(input_csv,output_excel)

    # merge_news_and_video_for_train()

    # merge_news_and_video_for_prediction()

    filter_and_split_p2_to_excel(input_csv,output_excel)