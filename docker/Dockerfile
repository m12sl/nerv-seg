FROM nvidia/cuda:7.5-cudnn4-devel
# Based on original tf dockerfile https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu

MAINTAINER Alex Ozerin <m12sl@yandex.ru>

ARG THEANO_VERSION=rel-0.8.2
ARG KERAS_VERSION=1.0.6


RUN apt-get autoclean && apt-get update && apt-get upgrade -y 
RUN apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        curl \
        git \
        wget \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        unzip \
        libhdf5-dev \
        zip \
        zlib1g-dev \
        libhdf5-dev \
        libyaml-dev \
        libjpeg-dev \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        libjpeg-dev \
        libopencv-dev \
        vim \
        python3 \
        python3-dev \
        python3-numpy \
        python3-scipy \
        python3-pip \
        python3-h5py \
        python3-yaml \
        swig \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3


RUN pip3 install --pre \
        tqdm \
        'numpy==1.11.0' \
        click \
        ipykernel \
        jupyter \
        matplotlib \
        cython \
        && \
    python3 -m ipykernel.kernelspec


# Upgrade six
ENV CUDA_ROOT /usr/local/cuda
ENV THEANO_FLAGS "floatX=float32,device=gpu,optimizer_including=cudnn,nvcc.fastmath=True,blas.ldflags='-L/usr/lib/openblas-base -lopenblas'"
RUN pip3 install git+git://github.com/Theano/Theano.git@${THEANO_VERSION}

RUN pip3 install git+git://github.com/fchollet/keras.git@${KERAS_VERSION}

# Mount volumes for actual
VOLUME ["/src", "/data", "/models"]

WORKDIR /
