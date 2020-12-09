FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates apt-transport-https gnupg2 curl && \
        curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub | apt-key add - && \
        echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
        echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list


# Install CUDA & CUDNN
ENV CUDA_VERSION 10.0.130
ENV CUDA_PKG_VERSION 10-0=$CUDA_VERSION-1

RUN apt-get update && apt-get install -y --no-install-recommends \
	cuda-cudart-$CUDA_PKG_VERSION \
    cuda-compat-10-0 \
    cuda-cusolver-10-0 \
    cuda-cublas-10-0 && \
    ln -s cuda-10.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

ENV CUDNN_VERSION 7.4.2.24
RUN apt-get update && apt-get install -y --no-install-recommends \
	libcudnn7=$CUDNN_VERSION-1+cuda10.0 \
	libcudnn7-dev=$CUDNN_VERSION-1+cuda10.0 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*


# Install python and corresponding packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
	libpython3-dev \
	build-essential \
	python3-pip \
	python3-setuptools && \
    rm -rf /var/lib/apt/lists/*


# Install numpy, tensorflow and openexr
RUN pip3 install --user tensorflow-gpu==1.13.2 \
    numpy==1.18.3 \
    wheel==0.35.1 \
    setuptools==50.3.2 \
    grpcio==1.27.2 \
    futures==3.1.1 \
    h5py==2.10.0


# Install openexr for python
RUN git clone https://github.com/jamesbowman/openexrpython.git
RUN apt-get update && apt-get install -y --no-install-recommends \
	libopenexr-dev \
	zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install openEXR==1.3.0
WORKDIR /openexrpython
RUN python3 setup.py install


# Running codes
VOLUME /data
VOLUME /codes
WORKDIR /codes
CMD python3 tester.py