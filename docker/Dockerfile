FROM nvidia/cuda:8.0-cudnn5-devel
# Based on original tf dockerfile https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu

MAINTAINER Alex Ozerin <m12sl@yandex.ru>

ARG THEANO_VERSION=rel-0.8.2
ARG KERAS_VERSION=1.0.7


RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        cmake \
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
        libavformat-dev \
        libjasper-dev \
        libjpeg-dev \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libtiff-dev \
        libpng-dev \
        libpq-dev \
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
        python3-tk \
        swig \
        yasm \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3


RUN pip3 install --pre \
        tqdm \
        click \
        ipykernel \
        jupyter \
        matplotlib \
        cython \
        flake8 \
        pep8 \
        && \
    python3 -m ipykernel.kernelspec

RUN wget https://github.com/Itseez/opencv/archive/3.0.0.zip \
    && unzip 3.0.0.zip \
    && mkdir /opencv-3.0.0/cmake_binary \
    && cd /opencv-3.0.0/cmake_binary \
    && cmake -DBUILD_TIFF=ON \
        -DBUILD_opencv_java=OFF \
        -DWITH_CUDA=OFF \
        -DENABLE_AVX=ON \
        -DWITH_OPENGL=ON \
        -DWITH_OPENCL=ON \
        -DWITH_IPP=ON \
        -DWITH_TBB=ON \
        -DWITH_EIGEN=ON \
        -DWITH_V4L=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
        -DPYTHON_EXECUTABLE=$(which python3) \
        -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -DPYTHON_PACKAGES_PATH=$(python3.5 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. \
    && make install \
    && rm /3.0.0.zip \
    && rm -r /opencv-3.0.0


# Upgrade six
ENV CUDA_ROOT=/usr/local/cuda \
    THEANO_FLAGS="floatX=float32,device=gpu,optimizer_including=cudnn,nvcc.fastmath=True,blas.ldflags='-L/usr/lib/openblas-base -lopenblas'"


RUN pip3 install git+git://github.com/Theano/Theano.git@${THEANO_VERSION} && \
    pip3 install git+git://github.com/fchollet/keras.git@${KERAS_VERSION}


# Mount volumes for actual
VOLUME ["/src", "/data", "/models"]

WORKDIR /