FROM hub.bitautotech.com/rc_nlp/cuda12.01-torch2.0.1:latest

WORKDIR /app

COPY . /app


# 安装其他依赖
RUN pip install --no-cache-dir Flask fastapi uvicorn tiktoken transformers_stream_generator einops accelerate wheel

EXPOSE 5060

#CMD bash -c "exec python yiche-news-summary/chat-server.py && python yiche-news-summary/infer-server.py"

CMD bash -c "exec python yiche_tags_classifier/app_p0-p2_tags_server.py"

