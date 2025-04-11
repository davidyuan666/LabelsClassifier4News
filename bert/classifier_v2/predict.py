# -*- coding: utf-8 -*-

import torch
from data_preprocess import load_json
from bert_multilabel_cls import BertMultiLabelCls
from transformers import BertTokenizer
from config import Config

label2idx = load_json(Config.label2idx_path)
idx2label = {idx: label for label, idx in label2idx.items()}

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

model = BertMultiLabelCls(hidden_size=Config.hidden_size, class_num=Config.class_num)
'''
预测的话是需要我们已经训练好的模型
'''
model.load_state_dict(torch.load(Config.save_model_path))
model.to(device)
model.eval()

def predict(texts):
    outputs = tokenizer(texts, return_tensors="pt", max_length=Config.max_len,
                        padding=True, truncation=True)
    logits = model(outputs["input_ids"].to(device),
                   outputs["attention_mask"].to(device),
                   outputs["token_type_ids"].to(device))
    logits = logits.cpu().tolist()
    result = []
    for sample in logits:
        pred_label = []
        for idx, logit in enumerate(sample):
            #设置阈值，也就是只要大于0.5的就输出,按道理可以设置topk和阈值
            if logit > 0.5:
                pred_label.append(idx2label[idx])
        result.append(pred_label)
    return result


if __name__ == '__main__':
    texts = ["中超-德尔加多扳平郭田雨绝杀 泰山2-1逆转亚泰", "今日沪深两市指数整体呈现震荡调整格局"]
    result = predict(texts)
    print(result)


