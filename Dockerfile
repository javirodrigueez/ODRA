FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

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

RUN apt-get update && apt-get install -y python3.13-venv
RUN python -m venv /opt/my_env
ENV PATH="/opt/my_env/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install vllm --extra-index-url https://download.pytorch.org/whl/cu130
