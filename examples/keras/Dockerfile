FROM nvcr.io/nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

LABEL maintainer="Lei Mao"

# Install package dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        wget \
        git \
        curl \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        zlib1g-dev \
        swig \
        vim \
        locales \
        locales-all
RUN apt-get clean

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

RUN cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    ln -s /usr/bin/pip3 pip && \
    pip install --upgrade pip setuptools

RUN pip install numpy==1.16.5 tensorflow-gpu==1.15.0 contextlib2==0.6.0 tqdm==4.36.1
RUN pip install tensorflow-datasets==1.2.0
