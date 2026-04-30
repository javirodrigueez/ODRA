FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    wget \
    unzip \
    curl \
    build-essential \
    g++ \
    cmake

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl git \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y python3.13 python3.13-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.13 /usr/bin/python

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13

# llama.cpp
## env vars
#ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64/stubs
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
## build llama.cpp
RUN git clone https://github.com/ggml-org/llama.cpp
RUN cd llama.cpp && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120" && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
    cmake --build build --config Release -j 24

RUN pip install backoff \
    openai \
    numpy \
    torch \
    transformers \
    unsloth \
    wandb

RUN cd llama.cpp && pip install -r requirements/requirements-convert_hf_to_gguf.txt

RUN pip install \
    omegaconf \
    vllm \
    langgraph \
    langchain \
    langchain_aws \
    langchain_huggingface \
    openai
