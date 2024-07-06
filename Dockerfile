# Base image
FROM 12.1.1-cudnn8-runtime-ubuntu22.04

# Use bash shell with pipefail option
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /

# Install system packages, clone repo, and cache models
ARG CIVITAI_TOKEN
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        wget ffmpeg libsm6 libxext6 git curl libgl1 libglib2.0-0 libgoogle-perftools-dev \
        python3.10-dev python3.10-tk python3-html5lib python3-apt python3-pip python3.10-distutils && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/kohya-ss/sd-scripts.git && \
    cd sd-scripts && \
    git checkout 25f961bc779bc79aef440813e3e8e92244ac5739
RUN mkdir -p /model_cache && \
    wget "https://civitai.com/api/download/models/292213?type=Model&format=SafeTensor&size=full&fp=fp16&token=${CIVITAI_TOKEN}" -O /model_cache/hyperRealism_30.safetensors

# Install Python dependencies (Worker Template)
COPY /requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Add src files (Worker Template)
ADD src /sd-scripts

COPY src/start.sh /start.sh

WORKDIR /sd-scripts

CMD /start.sh
