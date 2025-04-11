FROM hub.bitautotech.com/rc_nlp/cuda12.01-torch2.0.1:latest

WORKDIR /app

COPY . /app


# 安装其他依赖
RUN pip install --no-cache-dir Flask fastapi uvicorn tiktoken transformers_stream_generator einops accelerate
# 卸载ray包
#RUN pip uninstall ray
#RUN pip install wheel
#RUN pip install flash-attn --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
#RUN pip install -v flash-attn==2.1.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
# Clone and install the flash-attention package
#RUN cd flash-attention-main && \
#    pip install . && \
#    cd .. # Ensure we return to the /app directory


# Optionally install additional components. Uncomment if necessary.
# RUN cd flash-attention/csrc/layer_norm && pip install .
# RUN cd flash-attention/csrc/rotary && pip install .


EXPOSE 6001

#CMD bash -c "exec python yiche-news-summary/chat-server.py && python yiche-news-summary/infer-server.py"

CMD bash -c "exec python yiche_news/infer-server.py"

