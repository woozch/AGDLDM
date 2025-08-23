# Base image
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

# Environment settings
ENV TZ=Asia/Seoul
ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8

# Install essential packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    locales \
    sudo \
    wget \
    curl \
    ca-certificates

# Timezone Setting
RUN ls /usr/share/zoneinfo && \
    ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime \
    && echo "Asia/Seoul" >/etc/timezone

# Install basic packages
RUN apt-get update && apt-get install -y \
    build-essential \
    net-tools \
    curl \
    ca-certificates \
    git \
    libfreetype6-dev \
    libpng-dev \
    libzmq3-dev \
    pkg-config \
    software-properties-common \
    sudo \
    wget \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    zip \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# install python & R
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-numpy \
    python3-pip \
    python3-tk \
    r-base

# Optional: mpi packages
RUN sudo apt update && sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev

# install poetry
RUN conda install poetry -y
# terminal utility tools
RUN apt-get update && apt-get install -y tmux htop vim

# project name
ARG PROJECT=AGDLDM
WORKDIR /workspace/${PROJECT}

COPY . .
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction

# back to project directory
WORKDIR /workspace/${PROJECT}