# -*- coding: utf-8 -*-
"""
@Time ： 2024/2/29 14:55
@Auth ： David Yuan
@File ：flask_api_server_bk.py
@Institute ：BitAuto
"""

from flask import Flask, request, jsonify
from transformers import BertTokenizer
from bert_config import Config
from bert_model import YicheBertModel
import time
import csv
import os
from importlib import import_module
from textcnn_utils import build_dataset_from_api
import torch
import torch.nn as nn
from collections import defaultdict


app = Flask(__name__)
app.json.ensure_ascii = False

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_bert_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

def convert_to_labels(label_list, label_indices):
    return [label for label, index in zip(label_list, label_indices) if index == 1]

'''
tokenizer用基础模型的
'''
def load_bert_model(checkpoint_path,out_features):
    tokenizer = BertTokenizer.from_pretrained(Config.model_path)
    model = YicheBertModel(out_features).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=Config.LEARNING_RATE)
    model, optimizer = load_bert_ckp(checkpoint_path, model, optimizer)
    model.eval()
    return model,tokenizer


def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def predict_multi_tags(model, tokenizer, texts, label_list, device):
    all_predictions = []  # To store predictions with their texts

    for text in texts:  # Process each text individually
        processed_data = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=Config.MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        ids = processed_data['input_ids'].to(device, dtype=torch.long)
        mask = processed_data['attention_mask'].to(device, dtype=torch.long)
        token_type_ids = processed_data['token_type_ids'].to(device, dtype=torch.long)

        with torch.no_grad():
            outputs = model(ids, mask,token_type_ids)
            outputs = torch.sigmoid(outputs)

        label_indices = (outputs.cpu().numpy() > 0.5).astype(int).flatten()
        labels = [label_list[i] for i, label in enumerate(label_indices) if label == 1]

        all_predictions.append({"text": text, "labels": labels})  # Map text to its labels

    return all_predictions


def validate_input(data, required_fields):
    missing_fields = [field for field in required_fields if field not in data or data[field] is None]
    if missing_fields:
        return False, f"Missing fields: {', '.join(missing_fields)}"
    return True, ""


def load_textcnn_pretrained_model(model_name, dataset_dir, embedding_path):
    x = import_module('textcnn_pretrained_models.' + model_name)
    config = x.Config(dataset_dir, embedding_path)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    return model, config

def textcnn_predict(model, inputs_tensor_json_list):
    model.eval()
    results = []
    with torch.no_grad():
        for text_json_item in inputs_tensor_json_list:
            text = text_json_item['input_tensor']
            input_str = text_json_item['content']
            text = text.unsqueeze(0)  # Add the batch dimension
            text = text.unsqueeze(1)  # Add the channel dimension
            outputs = model(text)
            prediction = torch.max(outputs.data, 1)[1].cpu().numpy()
            print(f'content: {input_str}  ====>  prediction: {prediction}')
            results.append({'content': input_str,'prediction':str(prediction[0])})

    return results


'''
curl -X POST http://10.20.64.3:5060/tags/check -H "Content-Type: application/json" -d '{"contents": ["宝马M4彻底顶不住了？已有车源降16万！二手车商沉默了 宝马M4彻底顶不住了？已有车源降16万！二手车商沉默了","买第一辆车的时候，你踩了几个坑？ 姐妹们，无论是你已经买了第一辆车还是准备买，答应我，都要看完我这篇内容，下次不要再踩坑了。","深蓝S7：电感座驾，潮享未来 深蓝S7：电感座驾，潮享未来"], "request_id":"001", "content_ids":["100001","100002","100003"],"source_type":"news"}'

'''
@app.route('/tags/check', methods=['POST'])
def check_tags():
    data = request.get_json()
    required_fields = ['contents', 'request_id', 'content_ids','source_type']

    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

    vocab, contents = build_dataset_from_api(data['contents'], textcnn_config, use_word=False)

    if len(data['contents']) > 100:
        return jsonify({"error": "Too many contents, the maximum limit is 100."}), 413

    inputs_tensor_json_list = [
        {
            'input_tensor': torch.tensor(content[0], dtype=torch.long).to(textcnn_config.device),
            'content': content[1]
        } for content in contents
    ]

    final_results = []
    for model_key in ['p0', 'p1', 'p2']:
        model = textcnn_models[model_key]
        results = textcnn_predict(model, inputs_tensor_json_list)
        final_results.append({model_key:results})

    response = {
        "request_id": data['request_id'],
        "source_type": data['source_type'],
        "content_ids": data['content_ids'],
        "result": final_results
    }

    print('response', response)
    return jsonify(response)

'''
curl -X POST http://10.20.64.3:5060/tags/predict \
     -H "Content-Type: application/json" \
     -d '{
           "contents": ["宝马M4彻底顶不住了？已有车源降16万！二手车商沉默了 宝马M4彻底顶不住了？已有车源降16万！二手车商沉默了","买第一辆车的时候，你踩了几个坑？ 姐妹们，无论是你已经买了第一辆车还是准备买，答应我，都要看完我这篇内容，下次不要再踩坑了。","深蓝S7：电感座驾，潮享未来 深蓝S7：电感座驾，潮享未来"],
           "request_id": "33123131dsadsada",
           "content_ids":["100001","100002","100003"],
           "source_type": "news",
           "check_tag": "p0"
         }'
'''
@app.route('/tags/predict', methods=['POST'])
def predict_tags():
    data = request.get_json()
    required_fields = ['contents','content_ids', 'request_id', 'source_type', 'check_tag']

    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    contents = data['contents']
    check_tag = data['check_tag']

    if len(data['contents']) > 100:
        return jsonify({"error": "Too many contents, the maximum limit is 100."}), 413

    if check_tag == 'p2':
        check_tags = ['p2_1', 'p2_2']
    else:
        check_tags = [check_tag]

    content_label_map = defaultdict(list)

    for tag in check_tags:
        label_list = {
            'p0': ['谍照', '实车曝光', '配置曝光', '申报图', '预热', '新车上市', '预售', '发布亮相', '新车官图', '新车报价',
                           '新车到店', '新车解析'],
            'p1': ['车系品牌解读', '单车导购', '对比导购', '多车导购', '购车手册', '买车技巧', '评测导购', '汽车分享',
                           '试驾', '探店报价', '无解说车辆展示', '营销导购'],
            'p2_1': ['交通事故', '自燃', '维权事件', '车辆减配', '故障投诉', '车辆召回', '产能不足', '车辆首撞', '商家吐槽',
                             '爱车吐槽'],
            'p2_2': ['交通政策', '补贴政策', '汽车油价', '二手车限迁法规', '价格行情', '花边新闻', '销量新闻', '新闻聚合',
                             '人物观点', '行业评论', '汽车出口', '新能源新闻', '论坛峰会']
        }.get(tag, [])

        bert_model, bert_tokenizer = bert_models.get(tag, (None, None))
        if bert_model and bert_tokenizer:
            predicted_tags = predict_multi_tags(bert_model, bert_tokenizer, contents, label_list, device)
            for prediction in predicted_tags:
                content_label_map[prediction["text"]].extend(prediction["labels"])
        else:
            return jsonify({"error": f"Model for {tag} not found."}), 404

    all_predicted_tags = [{"content": text, "labels": list(set(labels))} for text, labels in content_label_map.items()]

    for prediction in all_predicted_tags:
        labels = prediction["labels"]
        label_hierarchy_paths = get_label_hierarchy_path(filtered_tag_hierarchy, labels)
        prediction["label_hierarchy_paths"] = label_hierarchy_paths

    response = {
        "request_id": data['request_id'],
        "source_type": data['source_type'],
        "check_tag": check_tag,
        "content_ids":data['content_ids'],
        "label_list": all_predicted_tags
    }

    print(f'response is: {response}')
    return jsonify(response)


def print_cuda_memory_usage(message_prefix=""):
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    blue_start = "\033[94m"
    blue_end = "\033[0m"
    print(
        f"{blue_start}{message_prefix} - Allocated Memory: {allocated:.2f} MiB, Reserved Memory: {reserved:.2f} MiB{blue_end}")


def get_label_hierarchy_path(filtered_tag_hierarchy, labels):
    label_paths = []
    for label in labels:
        for leaf_tag_id, hierarchy in filtered_tag_hierarchy.items():
            if label in hierarchy.values():
                path_elements = [hierarchy[tag_id] for tag_id in sorted(hierarchy.keys(), key=lambda x: int(x)) if
                                 tag_id in hierarchy and hierarchy[tag_id]]
                path = ",".join(path_elements)
                if path:
                    label_paths.append(path)
                break
    return label_paths


'''
加载所有的标签
'''
def load_tag_map():
    '''
    bash: warning: setlocale: LC_ALL: cannot change locale (zh_CN.UTF-8)
    Traceback (most recent call last):
      File "/app/yiche_tags_classifier/app_p0-p2_tags_server.py", line 319, in <module>
        filtered_tag_hierarchy = load_tag_map()
      File "/app/yiche_tags_classifier/app_p0-p2_tags_server.py", line 267, in load_tag_map
        with open(filename, mode='r', encoding='utf-8') as csvfile:
    FileNotFoundError: [Errno 2] No such file or directory: '/app/doc/all_tag_map.csv'
    '''
    filename = os.path.join(os.getcwd(),'yiche_tags_classifier','doc','all_tag_map.csv')
    tag_hierarchy = defaultdict(dict)

    with open(filename, mode='r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        for row in csv_reader:
            leaf_tag_id, tag_id_hierarchy, tag_name_hierarchy = row
            tag_ids = tag_id_hierarchy.split(',')
            tag_names = tag_name_hierarchy.split(',')
            for tag_id, tag_name in zip(tag_ids, tag_names):
                tag_hierarchy[leaf_tag_id][tag_id] = tag_name

    leaf_tag_list = {
        'p0': ['谍照', '实车曝光', '配置曝光', '申报图', '预热', '新车上市', '预售', '发布亮相', '新车官图', '新车报价',
               '新车到店', '新车解析'],
        'p1': ['车系品牌解读', '单车导购', '对比导购', '多车导购', '购车手册', '买车技巧', '评测导购', '汽车分享',
               '试驾', '探店报价', '无解说车辆展示', '营销导购'],
        'p2_1': ['交通事故', '自燃', '维权事件', '车辆减配', '故障投诉', '车辆召回', '产能不足', '车辆首撞', '商家吐槽',
                 '爱车吐槽'],
        'p2_2': ['交通政策', '补贴政策', '汽车油价', '二手车限迁法规', '价格行情', '花边新闻', '销量新闻', '新闻聚合',
                 '人物观点', '行业评论', '汽车出口', '新能源新闻', '论坛峰会']
    }

    filtered_tag_hierarchy = defaultdict(dict)
    for category, labels in leaf_tag_list.items():
        for label in labels:
            for leaf_tag_id, hierarchy in tag_hierarchy.items():
                if label in hierarchy.values():
                    filtered_tag_hierarchy[leaf_tag_id] = hierarchy
                    break

    for leaf_tag_id, hierarchy in filtered_tag_hierarchy.items():
        print(f"Leaf Tag ID: {leaf_tag_id}, Hierarchy: {hierarchy}")

    return filtered_tag_hierarchy


'''
对车展进行过滤
'''
def chezhan_regex():
    pass



'''
pip install scikit-learn  -i https://pypi.tuna.tsinghua.edu.cn/simple
CUDA_VISIBLE_DEVICES=1 python app_p0-p2_tags_server.py 
CUDA_VISIBLE_DEVICES=1 nohup python app_p0-p2_tags_server.py > flask_online_app.log 2>&1 &

https://md-to-pdf.fly.dev/

查看linux 开启的端口号:  netstat -tln
映射到A800: /data01/sharefile这个目录下
ark docker: /data/sharedpvc/david_tag_cls
'''
if __name__ == '__main__':
    filtered_tag_hierarchy = load_tag_map()
    print('CUDA Device Memory Usage Before Loading Models:')
    print_cuda_memory_usage("Before Loading Any Model")

    '''
    载入bert模型的p0,p1,p2-1,p2-2的多标签分类模型,加载容器共享网盘
    '''
    docker_share_base_path = '/data/sharedpvc/david_tag_cls'
    bert_checkpoints = {
        'p0': os.path.join(docker_share_base_path, 'bert_checkpoints_models', 'p0_checkpoints', 'checkpoint_epoch_8.pt'),
        'p1': os.path.join(docker_share_base_path, 'bert_checkpoints_models', 'p1_checkpoints', 'checkpoint_epoch_6.pt'),
        'p2_1': os.path.join(docker_share_base_path, 'bert_checkpoints_models', 'p2_1_checkpoints', 'checkpoint_epoch_13.pt'),
        'p2_2': os.path.join(docker_share_base_path, 'bert_checkpoints_models', 'p2_2_checkpoints', 'checkpoint_epoch_6.pt'),
    }

    out_features_dic = {
        'p0':13,
        'p1':12,
        'p2_1':10,
        'p2_2':13
    }

    '''
    加载上面的bert模型
    '''
    bert_models = {}
    for key, checkpoint_path in bert_checkpoints.items():
        out_features_num = out_features_dic[key]
        bert_model, bert_tokenizer = load_bert_model(checkpoint_path,out_features=out_features_num)
        bert_models[key] = (bert_model, bert_tokenizer)
        print_cuda_memory_usage(f"After Loading BERT Model {key}")


    '''
    开始加载textcnn二分类模型
    '''
    model_name = 'TextCNN'
    dataset_dir = '/data/sharedpvc/david_tag_cls/yiche-textcnn-cls-torch'
    embedding_name = 'yiche_embedding_sougounews_char.npz'
    textcnn_models = {}
    model_checkpoints = ['p0_textcnn_checkpoint_path', 'p1_textcnn_checkpoint_path', 'p2_textcnn_checkpoint_path']
    for i, checkpoint in enumerate(model_checkpoints):
        textcnn_model, textcnn_config = load_textcnn_pretrained_model(model_name, dataset_dir, embedding_name)
        checkpoint_path = getattr(textcnn_config, checkpoint)  # Dynamically get the path from configs
        textcnn_model.load_state_dict(torch.load(checkpoint_path))
        textcnn_model = textcnn_model.to(device)
        textcnn_model.eval()
        textcnn_models[f'p{i}'] = textcnn_model
        print_cuda_memory_usage(f"After Loading TextCNN Model p{i}")



    app.run(host='0.0.0.0', port=6002)
