FROM nvidia/cuda:7.5-cudnn4-devel
# Based on original tf dockerfile https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu

MAINTAINER Alex Ozerin <m12sl@yandex.ru>

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        curl \
        git \
        wget \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        python-numpy \
        python-pip \
        python-scipy \
        rsync \
        unzip \
        python3 \
        python3-dev \
        python3-numpy \
        python3-scipy \
        python3-pip \
        swig \
        zip \
        zlib1g-dev \
        libhdf5-dev \
        libyaml-dev \
        libjpeg-dev \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        libhdf5-dev \
        libjpeg-dev \
        libopencv-dev \
        vim \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        tqdm \
        click \
        ipykernel \
        jupyter \
        matplotlib \
        && \
    python -m ipykernel.kernelspec

RUN pip3 install --pre \
        tqdm \
        click \
        ipykernel \
        jupyter \
        matplotlib \
        cython \
        && \
    python3 -m ipykernel.kernelspec


# Install dependencies
RUN apt-get update && apt-get install -y \
  libhdf5-dev \
  python-h5py \
  python-yaml

# Upgrade six
RUN pip install --upgrade six

RUN pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
RUN pip3 install --upgrade --no-deps git+git://github.com/Theano/Theano.git


# Clone Keras repo and move into it
RUN cd /root && git clone https://github.com/fchollet/keras.git && cd keras && \
  # Install
  python setup.py install && \
  python3 setup.py install

RUN apt install -y python-opencv

# Mount volumes for actual
VOLUME ["/src", "/data", "/models"]

# Use Theano with GPU
ENV CUDA_ROOT /usr/local/cuda

RUN echo "[global]\n\
device = gpu\n\
floatX = float32\n" > ~/.theanorc

WORKDIR /