# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/2 17:52
@Auth ： David Yuan
@File ：configs.py
@Institute ：BitAuto
"""

# configs.py

DEFAULT_PORT = 6001  # 假设您通常使用这个端口
'''
对外访问的配置
'''
SERVER_URL = 'http://10.25.145.213'
ENDPOINT = '/infer'

'''
本地服务器的配置
'''
DEFAULT_PROMPT_VERSION = 'qwen-vllm'
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.8
DEFAULT_MAX_TOKENS = 500
TEST_COUNT=100
IS_DUAL=False
