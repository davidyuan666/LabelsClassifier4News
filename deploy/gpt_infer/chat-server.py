# -*- coding: utf-8 -*-
"""
@Time ： 2023/12/13 14:16
@Auth ： David Yuan
@File ：Chat_server.py
@Institute ：BitAuto
"""
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
import logging


log_path = '/data/sharedpvc/app.log'
# 检查日志文件路径是否存在，如果不存在就创建
if not os.path.exists(os.path.dirname(log_path)):
    os.makedirs(os.path.dirname(log_path))

# 使用 'a' 模式以确保日志内容被追加，而不是在每次运行脚本时被覆写
logging.basicConfig(filename=log_path, filemode='a', format='%(name)s - %(levelname)s - %(message)s')

RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
END = '\033[0m'
'''
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1    cuda=11.7的版本
参考 https://pytorch.org/get-started/previous-versions/
CUDA_VISIBLE_DEVICES=3 python Chat_server.py
'''
# os.environ['CUDA_VISIBLE_DEVICES'] = "5"
# app = FastAPI()
app = Flask(__name__)


'''
torch.float64 (或 torch.double): 双精度浮点数，提供更高的数值精度，但会增加内存消耗和计算时间。
torch.float32 (或 torch.float): 单精度浮点数，是深度学习中最常用的数据类型，提供良好的平衡在数值精度和性能之间。
torch.float16 (或 torch.half): 半精度浮点数，减少内存消耗，加速计算，但是牺牲了一定的数值精度，主要用于支持特定硬件的高性能计算。
torch.bfloat16: BFloat16浮点数，与float16类似，但提供了更宽的动态范围，这使得它在某些情况下比float16更有优势，尤其是在神经网络的训练过程中。
torch.int64 (或 torch.long): 64位整数，通常用于索引和计数。
torch.int32 (或 torch.int): 32位整数。
torch.int16 (或 torch.short): 16位整数。
torch.int8: 8位整数。
torch.uint8: 8位无符号整数。
torch.bool: 布尔类型。

Yi-34B-Chat-4bits(AWQ)
Yi-34B-Chat-8bits(GPTQ)
'''
def load_oyichat_model(model_path):
    try:
        model_name = 'oyichat'
        print("Starting to load the base model into GPU memory")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            use_flash_attention_2=True,
            device_map="auto",
            torch_dtype=torch.float16
        ).to('cuda').eval()
        print("Successfully loaded the model into GPU memory")
        return tokenizer, model,model_name
    except Exception as e:
        print(f'Exception occurred: {str(e)}')
        return None, None

'''
pip install transformers==4.32.0 accelerate tiktoken einops scipy transformers_stream_generator==0.0.4 peft deepspeed
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .
# 下方安装可选，安装可能比较缓慢。
# pip install csrc/layer_norm
# pip install csrc/rotary
'''
def load_qwen_model(model_path):
    try:
        print("Starting to load the qwen model into GPU memory")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # use bf16
        # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, bf16=True).to('cuda:1').eval()
        # use fp16
        # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, fp16=True).to('cuda:1').eval()
        # auto choose
        # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()

        # device_map = {0: ["encoder", "decoder.block.0"],
        #               1: ["decoder.block.1", "lm_head"]}

        # 使用bf16
        # model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, trust_remote_code=True, bf16=True).eval()

        # 使用fp16
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', trust_remote_code=True,
                                                     fp16=True).eval()

        # 自动选择
        # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()



        print("Successfully loaded the qwen model into GPU memory")
        return tokenizer, model
    except Exception as e:
        print(f'Exception occurred: {str(e)}')
        return None, None


'''
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2  
'''


def load_qwen_model_v2(model_path):
    try:
        print(f"Starting to load the base model into memory")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto").cuda()
        model = model.eval()

        # 打印模型加载到的设备
        print(f"Model is loaded on GPU: {next(model.parameters()).device}")

        print(f"Successfully loaded the model into memory")
        return tokenizer, model
    except Exception as e:
        print(e)
        return None, None


def load_qwen_model_v3(model_path):
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


def load_flashattn_qwen_model_v3(model_path):
    try:
        print(f"Starting to load the base model into memory")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import GenerationConfig
        from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            pad_token='<|extra_0|>',
            eos_token='<|endoftext|>',
            padding_side='left',
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            pad_token_id=tokenizer.pad_token_id,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        model.generation_config = GenerationConfig.from_pretrained(model_path, pad_token_id=tokenizer.pad_token_id)
        return tokenizer, model

    except Exception as e:
        print(e)
        return None, None

def load_vllm_qwen_model(model_path):
    try:
        print("Starting to load the llvm model into GPU memory")
        llm = LLM(model=model_path,trust_remote_code=True)
        print("Successfully loaded the llvm model into GPU memory")
        return llm
    except Exception as e:
        print(f'Exception occurred: {str(e)}')
        return None
'''
参考https://github.com/01-ai/Yi/blob/main/quantization/gptq/eval_quantized_model.py
https://github.com/01-ai/Yi/issues/206
Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
pip install autoawq
'''
def load_oyichat_quantization_model(model_path):
    try:
        print("Starting to load the quantized model into GPU memory")
        model_name = 'oyichat_quant'
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # attn_implementation="flash_attention_2",
            device_map = "auto",
            torch_dtype = torch.float16
        ).to('cuda').eval()

        print("Successfully loaded the quantized model into GPU memory")
        return tokenizer, model,model_name
    except Exception as e:
        print(f'Exception occurred: {str(e)}')
        return None, None

@app.post("/chat")
def chat():
    global model, tokenizer,vllm_model
    json_post_raw = request.json

    status_code = 1
    response = ""
    try:
        prompt = json_post_raw.get('prompt')
        version = json_post_raw.get('version', 'qwen-vllm')  # 如果没有提供version，则默认为'qwen-vllm'
        temperature = json_post_raw.get('temperature', 0.8)  # 如果没有提供temperature，则默认为0.7
        top_p = json_post_raw.get('top_p', 0.8)  # 如果没有提供top_p，则默认为0.9
        max_tokens = json_post_raw.get('max_tokens', 300)  # 如果没有提供max_tokens，则默认为50

        print(f'prompt is: {prompt}')
    except Exception as e:
        print(f'error is {str(e)}')
        logging.error(str(e))
        status_code = -1
        answer = {
            "status_code": status_code,
            "response": '',
            "history": ''
        }
        return answer

    try:
        start_time = time.time()

        if version == 'oyichat':
            messages = [
                {"role": "user", "content": prompt}
            ]
            input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=False,
                                                      return_tensors='pt')
            output_ids = model.generate(input_ids.to('cuda'))
            response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        if version == 'qwen':
            print('进入qwen进行推理')
            # inputs = tokenizer(prompt,
            #                    return_tensors='pt')
            # inputs = inputs.to(model.device)
            # pred = model.generate(**inputs)
            # response = tokenizer.decode(pred.cpu()[1], skip_special_tokens=True)
            response, history = model.chat(tokenizer, prompt, history=None)

        if version == 'qwen1.5':
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


        '''
        https://cloud.tencent.com/developer/article/2328353   可以并行推理
        离线批量推断
        prompts = [
            "prompt1","prompt2","prompt3"
        ]
        '''
        if version == 'qwen-vllm':
            print('进入qwen-vllm进行推理')

            # prompts = [prompt]
            # # 确保参数类型正确
            # sampling_params = SamplingParams(
            #     temperature=float(temperature),
            #     top_p=float(top_p),
            #     max_tokens=int(max_tokens)
            # )
            # outputs = llm.generate(prompts, sampling_params)
            # for output in outputs:
            #     generated_text = output.outputs[0].text
            #     print(f"Generated text: {generated_text!r}")
            #     response = generated_text


            '''
            开始调用vllm模块
            '''
            response, history = vllm_model.chat(query=prompt,
                                                history=None)

        end_time = time.time()
        print(f'cost is: {BLUE}{end_time - start_time}{END}')
        print(f'=====> response is: {response}')
    except Exception as e:
        logging.error(str(e))
        print(f'error is: {str(e)}')
        status_code = 0

    answer = {
        "status_code": status_code,
        "response": response,
        "history": ""
    }
    logging.info(f'answer is: {answer}')
    return answer


@app.post("/batch/chat")
def batch_chat():
    global model, tokenizer,model_name
    json_post_raw = request.json
    status_code = 1
    responses = []

    # prompt = json_post_raw.get('prompt')
    # version = json_post_raw.get('version')

    try:
        prompts = json_post_raw.get('prompts')
        if isinstance(prompts, str):
            prompts = json.loads(prompts)  # Parse the string to JSON if it's not already a dictionary
        prompt_base = prompts['prompt']  # The base prompt to prepend
        inputs = prompts['inputs']  # List of inputs
        messages_list = [
            {"role": "user", "content": f"{prompt_base}: {input_content}"} for input_content in inputs
        ]
    except Exception as e:
        print(f'Error parsing prompts: {str(e)}')
        logging.error(str(e))
        return {
            "status_code": -1,
            "responses": [],
            "history": ''
        }

    try:
        print(f'Messages list: {messages_list}')
        start_time = time.time()
        input_ids = tokenizer.apply_chat_template(conversation=messages_list, tokenize=True, add_generation_prompt=True,
                                                  return_tensors='pt')
        output_ids = model.generate(input_ids.to('cuda'))
        end_time = time.time()
        print(f'===>cost time: {end_time-start_time}')
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f'===> response: {response}')
        responses.append(response)

    except Exception as e:
        print(f'Error during response generation: {str(e)}')
        logging.error(str(e))
        status_code = 0

    return {
        "status_code": status_code,
        "responses": responses,
        "history": ""
    }

@app.get('/test')
def test_connetion():
    return {'message': 'Test Connection'}



'''
start loading Yi model chat server
CUDA_VISIBLE_DEVICES=2 python Chat_4090_server.py
CUDA_VISIBLE_DEVICES=0 python chat_vllm_server.py
CUDA_VISIBLE_DEVICES=0 python chat-server.py
pip install optimum
https://modelscope.cn/models/limoncc/Yi-34B-Chat-GGUF/summary
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen-14B-Chat',cached_dir='/data/david/models')
http://gitlab2.bitautotech.com/yuandw/yiche-news-summary-self-gpt.git
docker build -t my-flask-app:v1 .
curl http://10.25.145.149:6001/test
'''
if __name__ == "__main__":
    # tokenizer, model ,model_name= load_oyichat_model('/data/david/models/Yi-34B-Chat')  #A800
    # tokenizer, model,model_name = load_oyichat_model('/data01/davidyuan/models/Yi-34B-Chat')  #A100
    # tokenizer, model= load_qwen_model_v2('/data01/zn/qwen/Qwen-14B-Chat') # A100
    # tokenizer, model, model_name = load_qwen_model('/data/david/yiche-news-summary/models/Qwen-14B-Chat')  # 4090  transformers>=4.32.0
    # tokenizer, model, model_name = load_qwen_model('/data/david/yiche-news-summary/models/Qwen1.5-7B-Chat')  # 4090 transformers>=4.37.0
    # llm, model_name = load_vllm_qwen_model('/root/autodl-tmp/david/models/models/Qwen-7B-Chat/qwen/Qwen-7B-Chat')  # 4090  transformers>=4.32.0
    # tokenizer, model,model_name = load_oyichat_quantization_model('/data01/davidyuan/models/Yi-34B-Chat-4bits')  #A100  cost is: 279.90007638931274
    # tokenizer, model ,model_name= load_oyichat_quantization_model('/data01/davidyuan/models/Yi-34B-Chat-8bits')  # pip install optimum https://101.dev/t/autogptq-transformers/1145  cost is: 217.40508365631104
    # uvicorn.run(app, host='0.0.0.0', port=6001, workers=1)


    '''
    docker路径
    '''
    # tokenizer, model = load_qwen_model_v2('/data/david/models/Qwen-14B-Chat') # A800
    # tokenizer, model = load_qwen_model_v3('/data01/davidyuan/models/Qwen-14B-Chat')  #A100

    # tokenizer, model = load_qwen_model('/data/sharedpvc/model/Qwen-14B-Chat')
    # llm = load_vllm_qwen_model('/data/sharedpvc/model/Qwen-14B-Chat')
    # llm = load_vllm_qwen_model('/data/sharedpvc/model/models/qwen/Qwen-7B-Chat')


    '''
    由于ray漏洞问题，暂时不用vllm进行推理,替换为 FasterTransformer, TGI, vLLM, and FlashAttention.
    生成环境地址
    '''
    # model_path = "/data/sharedpvc/model/models/qwen/Qwen-7B-Chat"
    # model_path = '/data/sharedpvc/model/Qwen-14B-Chat'

    model_path = '/data/sharedpvc/model/Qwen-14B-Chat-Int4'

    '''
    vllm版本
    '''
    # vllm_model = vLLMWrapper(model_path,
    #                          quantization='gptq',
    #                          dtype="float16",
    #                          tensor_parallel_size=2)

    # vllm_model = vLLMWrapper(model_path,dtype="bfloat16",tensor_parallel_size=4)

    '''
    多gpu加载原生model
    '''
    tokenizer, model = load_qwen_model_v3(model_path)

    '''
    flashattn 加载model
    '''
    # tokenizer, model = load_flashattn_qwen_model_v3(model_path)


    app.run(host="0.0.0.0", port=6002)



