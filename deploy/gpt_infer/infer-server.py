# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/11 12:51
@Auth ： David Yuan
@File ：infer-server.py
@Institute ：BitAuto
"""


'''
常规库
'''
import requests
import re
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import logging
import os
import time
import torch
from fastapi import FastAPI, Request
from flask import Flask, request
import json, datetime, uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
'''
由于ray框架的安全原因，暂时不用vllm
'''
# from vllm import LLM, SamplingParams
from vllm_wrapper import vLLMWrapper



'''
print颜色
'''
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
END = '\033[0m'


'''
日志配置
'''
log_path = '/data/sharedpvc/app_inference.log'
# 检查日志文件路径是否存在，如果不存在就创建
if not os.path.exists(os.path.dirname(log_path)):
    os.makedirs(os.path.dirname(log_path))

# 使用 'a' 模式以确保日志内容被追加，而不是在每次运行脚本时被覆写
logging.basicConfig(filename=log_path, filemode='a', format='%(name)s - %(levelname)s - %(message)s')

'''
配置
'''
from config import (
    SERVER_URL, DEFAULT_PORT, ENDPOINT,
    DEFAULT_PROMPT_VERSION, DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P, DEFAULT_MAX_TOKENS,TEST_COUNT,
    IS_DUAL
)

'''
启动flask web服务
'''
app = Flask(__name__)


'''
旧的通过发送请求,已废弃
'''
def send_request_to_chat_server(prompt):
    global port_queue

    PORT = port_queue.popleft()
    full_url = f"{SERVER_URL}:{PORT}{ENDPOINT}"

    port_queue.append(PORT)

    # 构造请求载荷，使用 configs.py 中定义的默认值
    payload = {
        "prompt": prompt,
        "version": DEFAULT_PROMPT_VERSION,
        "temperature": str(DEFAULT_TEMPERATURE),
        "top_p": str(DEFAULT_TOP_P),
        "max_tokens": str(DEFAULT_MAX_TOKENS)
    }

    try:
        response = requests.post(full_url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Request failed with status code {response.status_code}"}
    except Exception as e:
        return {"error": f"An error occurred: {e}"}


class ModelBalancer:
    def __init__(self, model1, model2):
        self.models = [model1, model2]
        self.tokenizer = None  # 假设所有模型共用同一个tokenizer
        self.call_count = 0

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def chat(self, prompt, history=None):
        # 确定使用哪个模型
        current_model = self.models[self.call_count % len(self.models)]
        # 执行调用
        response, history = current_model.chat(self.tokenizer, prompt, history=history)
        # 更新计数器
        self.call_count += 1
        # 返回结果
        return response, history

class VllmModelBalancer:
    def __init__(self, model1, model2):
        self.models = [model1, model2]
        self.call_count = 0

    def chat(self, prompt):
        # 确定使用哪个模型
        current_model = self.models[self.call_count % len(self.models)]
        # 执行调用
        response, history = current_model.chat(query=prompt, history=None)

        # 更新计数器
        self.call_count += 1
        # 返回结果
        return response, history

'''
直接本地加载模型进行推理
'''
def infer_by_qwen_model(prompt):
    global vllm_model,tokenizer, model
    if model is None:
        print("Model is not loaded.")
    else:
        print("Model is loaded.")

    if vllm_model is None:
        print('vllm model is not loaded.')
    else:
        print('vllm model is loaded.')


    response = ""
    version = DEFAULT_PROMPT_VERSION
    status_code = 0
    try:
        if version == 'qwen':
            print('进入qwen进行推理')
            '''
            https://cloud.tencent.com/developer/article/2328353   可以并行推理
            离线批量推断
            prompts = [
                "prompt1","prompt2","prompt3"
            ]
            '''
            if IS_DUAL:
                print('=====> 进入双模型推理')
                # 现在可以通过model_balancer进行调用，它会在model1和model2之间轮询
                # response, history = model_balancer.chat(prompt, history=None)
            else:
                print('=====> 进入单模型推理')
                response, history = model.chat(tokenizer, prompt, history=None)

        if version == 'qwen-vllm':
            print('进入qwen-vllm进行推理')
            '''
            开始调用vllm模块
            '''
            if IS_DUAL:
                print('=====> 进入双模型vllm推理')
                # response, history = vllm_model_balancer.chat(prompt)
            else:
                print('=====> 进入单模型vllm推理')
                response, history = vllm_model.chat(query=prompt,history=None)


    except Exception as e:
        logging.error(str(e))
        status_code = -1
        response = str(e)

    answer = {
        "status_code": status_code,
        "response": response,
        "history": ""
    }

    return answer





'''
后处理
'''
def clean_text(text):
    # 去除特定的前缀词和修饰词
    prefixes = ['总结:', '概要:', '简写版:', '简写版:', '总结：', '概要：', '核心信息：', '核心信息:', '- ', ' -','主要内容','本文','要点']
    for prefix in prefixes:
        text = text.replace(prefix, '')

    # 如果文本包含"##$VideoFor"，直接返回空字符串或进行其他处理
    if '##$VideoFor' in text:
        return ''  # 或者其他适当的处理方式

    # 去除多余的空格和特殊字符
    # 注意：这里修改为去除字符串头尾的空白字符，而不是删除所有空格
    text = re.sub(r'\s+', ' ', text).strip()  # 将一个或多个空白字符替换成一个空格，并去除头尾空白

    return text


'''
业务方提供的java代码计算摘要个数
规则是:
汉字算一个，英文按照字符个数，句号，逗号当成一个，其他特殊字符全部去掉，阿拉伯数字也是按照字符个数来显示
'''
def count_words_characters(text):
    count = 0
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            count += 1
        elif char.isalpha() or char.isdigit():
            count += 1
        elif char in ['。', '，', '.', ',']:
            count += 1
    return count


def calculate_word_counts(text):
    try:
        if text is None:
            text = ""

        text = text.replace("！,", "！")
        text = text.replace("\"，\"", "")
        text = text.replace("\",\"", "")
        text = text.replace("；,", "；")
        text = text.replace("；，", "；")
        text = text.replace("。,", "。")
        text = text.replace("。，", "。")
        text = text.replace("。\"", "。")
        text = text.replace("\"", "")
        text = text.replace("！。", "！")
        text = text.replace("？。", "。")
        text = text.replace("&amp;", "")
        text = text.replace("&ldquo;", "")
        text = text.replace("&middot;", "")
        text = text.replace("&rdquo;", "")
        text = text.replace("&nbsp;", "")
        text = text.replace("&ndash;", "")
        count = count_words_characters(text)
        return count
    except Exception as e:
        return 0


def split_into_segments(text, max_length=1500):
    words = text.split()
    for i in range(0, len(words), max_length):
        yield ' '.join(words[i:i + max_length])


'''
内容后处理
'''
def remove_unwanted_patterns(part):
    numbering_pattern = re.compile(r'\d+\.\s+')
    clean_part = numbering_pattern.sub('', part).strip()
    clean_part = re.sub(r'\s+', ' ',
                        clean_part)  # Replace all whitespace characters with a single space
    for keyword in ['简写内容', ':', '：', '主要信息', '标题', '正文', '汽车新闻编辑', '主要内容',
                    '要点', '如下 ','摘要 ','摘要','此外,','总之,','然而,','此外，','总之，','然而，',
                    '同时,','同时，','摘要,','摘要，','以下为提取的','最后,','最后，','另外,','另外，',
                    '好的，','好的,','以下是提取的','以下是一份最这篇文章的提取','以下是','以下是该文章的',
                    '以下是该内容的','总的来说,','总的来说，','以下是从文章中提取的','文章指出，','文章指出,',
                    '以下是对文本的提取','以下是分好的句子','以下是从新闻中提取的','以下是提取的','根据文章提供的信息，','根据文章提供的信息,',
                    '提取','以下是对原文的提取','一、','二、','根据提供的信息，以下是可以从文章中提取的主要',
                    '以下是','以下是提取','以下是文章的摘要','以下是提取','以下是对该新闻的的提取 ','以下是 ','以下是',
                    '以下是一些','三、','四、','五、','','根据文章，以下是一些 ','2）','3）','4）','5）','根据您提供的内容，以下是 ',
                    '以下是关于','以下是一些摘要','内容提要','以下是该新闻的主要','以下是提取','以下是对所给信息的提取',
                    '以下是','以下是将这些内容分隔成多句话的版本','以下是对给定内容进行分句并使其通顺且不增加字数的结果','以下为提取',
                    '主要介绍的内容包括以下几点','以下是该提取的','文章的可以总结','提取 ','提取','以下是最主要的信息摘要',
                    '以下是','根据提供的信息，','文章 ','提取','这篇提取了以下','以下是将内容摘要分成不止一句话的结果','信息提取 ',
                    '以下是这篇文章的','这篇的是','以下是提取的','以下是从新闻中提取的','文章的','以下为新闻的','该文章提到了以下关键信息',
                    '该文章的 '
                    ]:
        clean_part = clean_part.replace(keyword, '')
    return clean_part

'''
处理原始文本到多个文本
'''
def process_res_content_to_splits(content):
    print(f'Original: {content}')
    res_content = clean_text(content)
    split_content = res_content.split('。')
    formatted_split_content = []

    i = 1
    print(f'开始分段处理....')
    for part in split_content:
        if not part or i > 5:
            continue

        print('################################################################################')
        print(f'待处理的单条信息: {RED}{part}{END}')


        part_without_numbering = remove_unwanted_patterns(part)

        '''
        如果单条信息太长
        '''
        if len(part_without_numbering) > 60:
            print(f'本条信息长度大于60，处理的prompt是:{text_split_prompt}')
            prompt = f"{text_split_prompt}:{part_without_numbering}"
            # res = send_request_to_chat_server(prompt)
            '''
            本地推理
            '''
            res = infer_by_qwen_model(prompt)
            if res and 'response' in res:
                segment_res_content = res['response']
                segments = segment_res_content.split('。')
                for segment in segments:
                    if i > 5:
                        break
                    if segment:
                        segment_without_numbering = remove_unwanted_patterns(segment)
                        formatted_part = f'{i}.{segment_without_numbering.strip()}'
                        formatted_split_content.append(formatted_part)
                        print(f'=>{BLUE}{formatted_part}{END}')
                        i += 1
        else:
            formatted_part = f'{i}.{part_without_numbering.strip()}'
            formatted_split_content.append(formatted_part)
            print(f'{BLUE}{formatted_part}{END}')
            i += 1

    '''
    返回带有序号的总结文本
    '''
    return formatted_split_content



'''
优化的点:
批量化处理prompt
对输出的部分数字进行校正，fix的error:
(1)数字提取有误，包括价格、里程等 我看像12-45万元这种有“-”的，都把“-”去掉了，导致车的价格变成了1245万元 ,还有是像【山海平台2.0】、【e平台1.0】这种也会漏掉小数点和前面的数字，这块审核时候容易看不出来，所以数字相关的最好能和正文一致
(2)要点完全重复：同一句话出现两次、多次
(3)出现符号或网址等信息
(4)个别语句中出现【要点】【主要内容】等导致不通顺
(5)要点提取仅针对一个方向
'''
def inference_by_batch(input_contents):
    dic = {}
    try:
        response_list = []
        start_time = time.time()
        for input_content in input_contents:
            print(f'========>开始处理{input_content}')
            word_count = calculate_word_counts(input_content)
            if word_count > 2500:
                print('本文超过2500字开始长文本进行裁剪总结')
                content_segments = list(split_into_segments(input_content, 500))
                final_content = ""
                for segment_content in content_segments:
                    prompt = f"{large_text_prompt}:{segment_content}"
                    # result = send_request_to_chat_server(prompt)
                    '''
                    本地推理
                    '''
                    result = infer_by_qwen_model(prompt)
                    '''
                    Generated text: "<|im_end|>\n<|im_start|>'t\n试车员是汽车制造业中的重要一环，"
                    '''
                    print(result)
                    if result and 'response' in result:
                        res_content = result['response']
                        final_content += res_content + " "

                input_content = final_content

            if word_count < 150:
                print('本文小于150字开始扩展')
                expand_prompt = f"{expand_text_prompt}:{input_content}"
                # result = send_request_to_chat_server(expand_prompt)
                result = infer_by_qwen_model(expand_prompt)
                if result and 'response' in result:
                    input_content = result['response']

            '''
            正常性总结
            '''
            prompt = f"{default_prompt}:{input_content}"
            # res = send_request_to_chat_server(prompt)
            '''
            本地推理
            '''
            res = infer_by_qwen_model(prompt)
            if res and 'response' in res:
                response_list.append(res['response'])
        '''
        开始后处理
        '''
        end_time = time.time()
        first_cost_time = end_time-start_time

        print('==========>后处理step1')
        start_time = time.time()
        for res_content in response_list:
            formatted_split_content = process_res_content_to_splits(res_content)
            '''
            fix bugs
            对输出的部分数字进行校正，fix的error:
            (1)数字提取有误，包括价格、里程等 我看像12-45万元这种有“-”的，都把“-”去掉了，导致车的价格变成了1245万元 ,还有是像【山海平台2.0】、【e平台1.0】这种也会漏掉小数点和前面的数字，这块审核时候容易看不出来，所以数字相关的最好能和正文一致
            (2)要点完全重复：同一句话出现两次、多次
            (3)出现符号或网址等信息
            (4)个别语句中出现【要点】【主要内容】等导致不通顺
            (5)要点提取仅针对一个方向
            '''

            print('====>后处理step2')
            new_formatted_split_content = []
            for formatted_part in formatted_split_content:
                '''
                如果内容太短的话也不需要处理
                '''
                if len(formatted_part) < 10:
                    continue

                formatted_part = formatted_part.replace('- ', '').replace(' - ', '').replace(' -', '')
                print(f'===>{GREEN} {formatted_part}{END}')
                new_formatted_split_content.append(formatted_part)

            dic[res_content]=new_formatted_split_content


        end_time = time.time()
        print(f'第一轮处理 cost time{BLUE} {first_cost_time}{END}')
        print(f'第二轮处理 cost time{BLUE} {end_time-start_time}{END}')

        return dic
    except Exception as e:
        print(str(e))
        return None


'''
推理后处理
'''
def inference(input_content):
    response = ''
    try:
        response_list = []
        start_time = time.time()
        print(f'========>开始处理{input_content}')
        word_count = calculate_word_counts(input_content)
        if word_count > 2500:
            print('本文超过2500字开始长文本进行裁剪总结')
            content_segments = list(split_into_segments(input_content, 500))
            final_content = ""
            for segment_content in content_segments:
                prompt = f"{large_text_prompt}:{segment_content}"
                # result = send_request_to_chat_server(prompt)
                '''
                本地推理
                '''
                result = infer_by_qwen_model(prompt)
                '''
                Generated text: "<|im_end|>\n<|im_start|>'t\n试车员是汽车制造业中的重要一环，"
                '''
                print(result)
                if result and 'response' in result:
                    res_content = result['response']
                    final_content += res_content + " "

            input_content = final_content

        if word_count < 150:
            print('本文小于150字开始扩展')
            expand_prompt = f"{expand_text_prompt}:{input_content}"
            # result = send_request_to_chat_server(expand_prompt)
            result = infer_by_qwen_model(expand_prompt)
            if result and 'response' in result:
                input_content = result['response']

        '''
        正常性总结
        '''
        prompt = f"{default_prompt}:{input_content}"
        # res = send_request_to_chat_server(prompt)
        '''
        本地推理
        '''
        res = infer_by_qwen_model(prompt)
        if res and 'response' in res:
            response_list.append(res['response'])


        '''
        开始后处理
        '''
        end_time = time.time()
        first_cost_time = end_time - start_time

        print('==========>后处理step1')
        start_time = time.time()
        for res_content in response_list:
            formatted_split_content = process_res_content_to_splits(res_content)
            '''
            fix bugs
            对输出的部分数字进行校正，fix的error:
            (1)数字提取有误，包括价格、里程等 我看像12-45万元这种有“-”的，都把“-”去掉了，导致车的价格变成了1245万元 ,还有是像【山海平台2.0】、【e平台1.0】这种也会漏掉小数点和前面的数字，这块审核时候容易看不出来，所以数字相关的最好能和正文一致
            (2)要点完全重复：同一句话出现两次、多次
            (3)出现符号或网址等信息
            (4)个别语句中出现【要点】【主要内容】等导致不通顺
            (5)要点提取仅针对一个方向
            '''

            print('====>后处理step2')
            new_formatted_split_content = []
            for formatted_part in formatted_split_content:
                '''
                如果内容太短的话也不需要处理
                '''
                if len(formatted_part) < 10:
                    continue

                formatted_part = formatted_part.replace('- ', '').replace(' - ', '').replace(' -', '')
                print(f'===>{GREEN} {formatted_part}{END}')
                new_formatted_split_content.append(formatted_part)

            response = new_formatted_split_content

        end_time = time.time()
        print(f'第一轮处理 cost time{BLUE} {first_cost_time}{END}')
        print(f'第二轮处理 cost time{BLUE} {end_time - start_time}{END}')

        return response
    except Exception as e:
        print(str(e))
        return None




'''
推理服务
'''
@app.post("/infer")
def infer():
    json_post_raw = request.json
    status_code = 1
    try:
        content = json_post_raw.get('content')
        print(f'input content is: {content}')
    except Exception as e:
        logging.error(str(e))
        res = {
            "status_code": -1,
            "response": str(e)
        }
        return res

    try:
        start_time = time.time()
        '''
        开始推理
        '''
        response = inference(content)
        '''
        把推理结果写入
        '''
        # output_file_path = f'yiche_output_qwen7b.csv'
        # with (open(output_file_path, 'a', newline='', encoding='utf-8') as csvfile):
        #     writer = csv.writer(csvfile)
        #     writer.writerow(['Content', 'ModelResponse'])
        #     for key, value in dic.items():
        #         response_content = ''
        #         for value_str in value:
        #             response_content += value_str + '\n'
        #         writer.writerow([key, response_content])

        # print('写入完毕')

        '''
        后处理返回的值
        '''
        end_time = time.time()
        logging.info(f'cost is: {BLUE}{end_time - start_time}{END}')
        print(f'cost is: {BLUE}{end_time - start_time}{END}')

        response_content = ''
        for value_str in response:
            response_content += value_str + '\n'

        res = {
            "status_code": status_code,
            "response": response_content,
        }
        return res


    except Exception as e:
        logging.error(str(e))
        res = {
            "status_code": -1,
            "response": str(e)
        }
        return res



@app.get('/test')
def test_connetion():
    return {'message': 'Test Connection for inference'}


# def load_qwen_model_v1(model_path):
#     try:
#         print("Starting to load the base model into memory")
#         tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#         # 自动选择
#         model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).cuda().eval()
#
#         print("Successfully loaded the model into memory")
#         return tokenizer, model
#     except Exception as e:
#         print(e)
#         return None, None

'''
加载多个gpuss
'''


def reserve_memory(device):
    total_memory = torch.cuda.get_device_properties(device).total_memory
    reserve_memory = total_memory * 0.2  # Reserve 20% of the total memory
    buffer = torch.cuda.memory_allocate(device, reserve_memory)
    torch.cuda.memory_deallocate(buffer)  # Immediately deallocate to just limit available memory

# def load_qwen_model_gpus(model_path):
#     try:
#         print("Starting to load the base model into GPUs...")
#         if torch.cuda.is_available():
#             device = torch.device('cuda')
#             num_gpus = torch.cuda.device_count()
#             print(f"Available GPUs: {num_gpus}")
#             for i in range(num_gpus):
#                 reserve_memory(i)  # Reserve memory on each GPU
#         else:
#             device = torch.device('cpu')
#             num_gpus = 0
#             print("CUDA is not available, using CPU instead.")
#
#         # Load tokenizer and model with trust_remote_code=True to allow custom code execution
#         tokenizer = AutoTokenizer.from_pretrained(model_path, device_map='auto', trust_remote_code=True)
#         model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', trust_remote_code=True)
#
#         # Move the model to the appropriate device
#         model.to(device)
#
#         # If more than one GPU is available, wrap the model with DataParallel
#         if num_gpus > 1:
#             model = torch.nn.DataParallel(model)
#             print("Model has been wrapped in DataParallel for multi-GPU support.")
#
#         print(f"Successfully loaded the model on {'multiple GPUs' if num_gpus > 1 else 'a single GPU'}.")
#         return tokenizer, model
#
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None, None

'''
自己的版本
'''
def load_qwen_model_gpus(model_path):
    try:
        print("Starting to load the base model into GPUs...")
        if torch.cuda.is_available():
            device = torch.device('cuda')
            num_gpus = torch.cuda.device_count()
            print(f"Available GPUs: {num_gpus}")
        else:
            device = torch.device('cpu')
            num_gpus = 0
            print("CUDA is not available, using CPU instead.")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

        # Move the model to the appropriate device
        model.to(device)
        torch.cuda.empty_cache()  # Clear cache after moving model

        # If more than one GPU is available, wrap the model with DataParallel
        if num_gpus > 1:
            model = torch.nn.DataParallel(model)
            print("Model has been wrapped in DataParallel for multi-GPU support.")

        print(f"Successfully loaded the model on {'multiple GPUs' if num_gpus > 1 else 'a single GPU'}.")
        return tokenizer, model

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None



import torch
from transformers import AutoModelForCausalLM
from accelerate import dispatch_model


def _device_map(num_gpus, num_layers):
    per_gpu_layers = (num_layers + 2) / num_gpus

    device_map = {
        'transformer.wte': 0,
        'transformer.ln_f': 0,
        'lm_head': num_gpus-1
    }

    used = 1
    gpu_target = 0
    for i in range(num_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0 if gpu_target < num_gpus-1 else 1
        assert gpu_target < num_gpus
        device_map[f'transformer.h.{i}'] = gpu_target
        used += 1

    return device_map

'''
官方的版本
'''
def load_qwen_model_on_gpus(model_name_or_path, num_gpus: int = 2):
    try:
        print("Starting to load the base model into GPUs...")
        num_devices = torch.cuda.device_count()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if num_gpus == 1:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto',
                                                         trust_remote_code=True).eval()
        elif 1 < num_gpus <= num_devices:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='cpu',
                                                         trust_remote_code=True).eval()
            num_layers = model.config.num_hidden_layers
            device_map = _device_map(num_gpus, num_layers)
            print(device_map)
            print(f"Successfully loaded the model on {'multiple GPUs' if num_gpus > 1 else 'a single GPU'}.")
            model = dispatch_model(model, device_map=device_map)
        else:
            raise KeyError

        return tokenizer, model
    except Exception as e:
        print(str(e))
        return None,None



import torch.distributed as dist
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def model_to_ddp(model_path, rank):
    setup(rank, 2)  # world_size is 2 since we use 2 GPUs
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(rank)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    return tokenizer, ddp_model

def load_qwen_model_v1(model_path, device_id=None):
    try:
        print("Starting to load the base model into memory")
        # 判断是否指定了设备编号，如果没有，则自动选择
        if device_id is not None and torch.cuda.is_available():
            device = torch.device(f'cuda:{device_id}')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

        # 将模型加载到指定的设备上
        model.to(device).eval()

        print(f"Successfully loaded the model into memory on {device}")
        return tokenizer, model
    except Exception as e:
        print(e)
        return None, None

def load_qwen_model_v2(model_path):
    try:
        print("Starting to load the base model into memory")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

        if torch.cuda.is_available():
            # 确保CUDA可用
            # device = torch.device("cuda:0")  # 将模型加载到第一个GPU
            # model.to(device)  # 移动模型到指定的设备
            if torch.cuda.device_count() > 1:
                # 如果有多于一个GPU可用，使用DataParallel来使用它们
                model = torch.nn.DataParallel(model, device_ids=[0, 1])  # 指定在GPU 0和GPU 1上运行
                print(f"Model is loaded on GPUs: {[device for device in range(torch.cuda.device_count())]}")
            else:
                print("Only one GPU found. The model will run on a single GPU.")
        else:
            print("CUDA is not available. Model will run on CPU.")

        model = model.eval()  # 将模型设置为评估模式

        print("Successfully loaded the model into memory")
        return tokenizer, model
    except Exception as e:
        print(e)
        return None, None

if __name__ == '__main__':
    large_text_prompt = '你是一个汽车新闻编辑,将下面内容减少字数但提取主要信息'  # 内容太多
    expand_text_prompt = '稍微扩充一点内容并总结为序号显示，不多于5条'  # 内容太少
    default_prompt = '你是一个汽车新闻编辑,请把下面内容提取主要信息,保持数字和小数点不变'  # 正常内容
    text_split_prompt = '你是一个汽车新闻编辑,把下面内容分成不止一句话并让句子通顺，但是字数不能增多'  # 单条句子太长

    tokenizer = None
    model = None
    '''
    加载模型路径
    '''
    # model_path = "/data/sharedpvc/model/models/qwen/Qwen-7B-Chat"
    model_path = '/data/sharedpvc/model/Qwen-14B-Chat'
    # model_path = '/data/sharedpvc/model/Qwen-14B-Chat-Int4'
    '''
    加载qwen模型 多个gpus
    '''
    # tokenizer, model = load_qwen_model_gpus(model_path)  # 自己版本

    # tokenizer, model  = load_qwen_model_on_gpus(model_path) # 官方版本
    # model_to_ddp(model_path,)

    '''
    单gpu加载模型
    '''
    # tokenizer, model1 = load_qwen_model_v1(model_path,device_id=0)
    # tokenizer2, model2 = load_qwen_model_v1(model_path,device_id=1)
    # # 使用示例
    # model_balancer = ModelBalancer(model1, model2)
    # model_balancer.set_tokenizer(tokenizer)

    '''
    加载vllm加速模型
    '''
    vllm_model = vLLMWrapper(model_path,device_id=0,dtype="bfloat16",tensor_parallel_size=2)  #缺少share memory设置
    # vllm_model_2 = vLLMWrapper(model_path,device_id=1, dtype="bfloat16", tensor_parallel_size=1)
    #
    # vllm_model_balancer = VllmModelBalancer(vllm_model_1,vllm_model_2)
    app.run(host="0.0.0.0", port=DEFAULT_PORT)
