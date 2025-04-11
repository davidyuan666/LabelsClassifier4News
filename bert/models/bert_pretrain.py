# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/18 10:42
@Auth ： David Yuan
@File ：bitauto_pretrain_bert.py
@Institute ：BitAuto
"""
from tqdm import tqdm
import random
import torch
import torch.nn as nn
from torch.nn.init import normal_
from torch.utils.data import DataLoader
import os
import logging
from transformers import BertTokenizer
import math
import json
from torch.nn import functional as F

'''
需求，预训练一个易车领域的BERT模型
如何利用Mask LM和NSP这两个任务来训练BERT模型进行介绍。通常，你既可以通过MLM和NSP任务来从头训练一个BERT模型，
当然也可以在开源预训练模型的基础上再次通过MLM和NSP任务来在特定语料中进行追加训练，以使得模型参数更加符合这一场景。
参考 https://github.com/moon-hotel/BertWithPretrained/blob/main/README-zh-CN.md
'''
'''
read_wiki2 读取wiki2数据
'''
def read_wiki2(filepath=None):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    paragraphs = []
    for line in tqdm(lines, ncols=80, desc=" ## 正在读取原始数据"):
        if len(line.split(' . ')) >= 2:
            paragraphs.append(line.strip().lower().split(' . '))
    random.shuffle(paragraphs)
    return paragraphs

'''
read_songci 读取诗词数据
'''
def read_songci(filepath=None):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    paragraphs = []
    for line in tqdm(lines, ncols=80, desc=" ## 正在读取原始数据"):
        if "□" in line or "……" in line:
            continue
        if len(line.split('。')) >= 2:
            paragraphs.append(line.strip().split('。')[:-1])
    random.shuffle(paragraphs)
    return paragraphs

'''
read_custom 读取易车自定义格式的数据
'''
def read_custom(filepath=None):
    """读取易车自定义格式的数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    paragraphs = []
    for line in tqdm(lines, ncols=80, desc=" ## 正在读取易车数据"):
        # 假设每行是一个JSON，包含文本内容
        try:
            data = json.loads(line.strip())
            text = data.get('content', '').split('。')
            if len(text) >= 2:  # 确保文本可以分成至少两句话
                paragraphs.append(text)
        except:
            continue
    random.shuffle(paragraphs)
    return paragraphs

'''
BertConfig 配置
'''
class BertConfig:
    def __init__(self):
        self.vocab_size = 21128  # 词表大小
        self.hidden_size = 768  # 隐藏层维度
        self.num_hidden_layers = 12  # Transformer编码器层数
        self.num_attention_heads = 12  # 注意力头数
        self.intermediate_size = 3072  # FFN中间层维度
        self.hidden_dropout_prob = 0.1  # 隐藏层dropout概率
        self.attention_probs_dropout_prob = 0.1  # 注意力dropout概率
        self.max_position_embeddings = 512  # 最大位置编码长度
        self.type_vocab_size = 2  # segment类型数量
        self.initializer_range = 0.02  # 参数初始化范围
        self.layer_norm_eps = 1e-12  # LayerNorm epsilon值
        self.pad_token_id = 0  # padding token的id

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_token_id=0, initializer_range=0.02):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self._reset_parameters(initializer_range)

    def forward(self, input_ids):
        return self.embedding(input_ids)

    def _reset_parameters(self, initializer_range):
        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=initializer_range)

class PositionalEmbedding(nn.Module):
    def __init__(self, hidden_size, max_position_embeddings=512, initializer_range=0.02):
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self._reset_parameters(initializer_range)

    def forward(self, position_ids):
        return self.embedding(position_ids)

    def _reset_parameters(self, initializer_range):
        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=initializer_range)

class SegmentEmbedding(nn.Module):
    def __init__(self, type_vocab_size, hidden_size, initializer_range=0.02):
        super(SegmentEmbedding, self).__init__()
        self.embedding = nn.Embedding(type_vocab_size, hidden_size)
        self._reset_parameters(initializer_range)

    def forward(self, token_type_ids):
        return self.embedding(token_type_ids)

    def _reset_parameters(self, initializer_range):
        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=initializer_range)

class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            pad_token_id=config.pad_token_id,
            initializer_range=config.initializer_range)

        self.position_embeddings = PositionalEmbedding(
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            initializer_range=config.initializer_range)

        self.token_type_embeddings = SegmentEmbedding(
            type_vocab_size=config.type_vocab_size,
            hidden_size=config.hidden_size,
            initializer_range=config.initializer_range)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention heads {config.num_attention_heads}")
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids
        )
        
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask
        )
        
        pooled_output = self.pooler(encoder_outputs)
        return encoder_outputs, pooled_output

class BertForPreTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.config = config

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, mlm_labels=None, nsp_labels=None):
        sequence_output, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )
        
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output)
        
        total_loss = None
        if mlm_labels is not None and nsp_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                mlm_labels.view(-1)
            )
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2),
                nsp_labels.view(-1)
            )
            total_loss = masked_lm_loss + next_sentence_loss
            
        return total_loss if total_loss is not None else (prediction_scores, seq_relationship_score)


'''
训练BERT模型
'''
def train_bert(model, train_iter, optimizer, num_epochs, device, save_dir):
    """训练BERT模型"""
    model.train()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_iter, desc=f"Training epoch {epoch}")):
            b_input_ids, b_token_type_ids, b_attention_mask, b_mlm_labels, b_nsp_labels = [
                t.to(device) for t in batch]
            
            optimizer.zero_grad()
            loss = model(
                input_ids=b_input_ids,
                attention_mask=b_attention_mask,
                token_type_ids=b_token_type_ids,
                mlm_labels=b_mlm_labels,
                nsp_labels=b_nsp_labels
            )
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # 每100个batch打印一次loss
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_iter)
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        
        # 保存最好的模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(save_dir, 'best_yiche_bert.pt'))

def main():
    # 设置随机种子
    torch.manual_seed(42)
    random.seed(42)
    
    # 配置
    model_config = ModelConfig()
    bert_config = BertConfig()
    
    # 创建保存目录
    os.makedirs(model_config.model_save_dir, exist_ok=True)
    os.makedirs(model_config.logs_save_dir, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        filename=os.path.join(model_config.logs_save_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 数据加载器
    data_loader = LoadBertPretrainingDataset(
        vocab_path=model_config.vocab_path,
        tokenizer=BertTokenizer.from_pretrained(
            model_config.pretrained_model_dir).tokenize,
        batch_size=model_config.batch_size,
        max_sen_len=model_config.max_sen_len,
        max_position_embeddings=model_config.max_position_embeddings,
        pad_index=model_config.pad_index,
        is_sample_shuffle=model_config.is_sample_shuffle,
        random_state=model_config.random_state,
        data_name=model_config.data_name,
        masked_rate=model_config.masked_rate,
        masked_token_rate=model_config.masked_token_rate,
        masked_token_unchanged_rate=model_config.masked_token_unchanged_rate
    )
    
    # 加载数据
    train_iter, val_iter, test_iter = data_loader.load_train_val_test_data(
        train_file_path=model_config.train_file_path,
        val_file_path=model_config.val_file_path,
        test_file_path=model_config.test_file_path
    )
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForPreTraining(bert_config).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # 训练
    num_epochs = 10  # 可以根据需要调整
    train_bert(model, train_iter, optimizer, num_epochs, device, model_config.model_save_dir)
    
    logging.info("Training completed!")

if __name__ == '__main__':
    main()