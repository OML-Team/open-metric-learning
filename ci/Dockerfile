ARG RUNTIME=cpu


FROM nvidia/cuda:11.1.1-devel-ubuntu18.04 AS gpu_base

ENV TORCH_POSTFIX=cu111

FROM ubuntu:18.04 AS cpu_base
ENV TORCH_POSTFIX=cpu


FROM ${RUNTIME}_base

### Basic libs
RUN apt-get -y update
RUN apt-get -y install git wget cmake openssh-server \
    libsm6 libxext6 libfontconfig1 libxrender1 libgl1-mesa-dev \
    libglib2.0-0 libgtk2.0-dev libxrender-dev libjpeg-dev zlib1g-dev

### Python
RUN apt install -y python3.8-dev python3.8-distutils python3-pip
RUN ln -sf /usr/bin/python3.8 /usr/bin/python && ln -sf /usr/bin/python3.8 /usr/bin/python3
ENV LD_LIBRARY_PATH "/usr/local/lib:$LD_LIBRARY_PATH"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN pip3 install --upgrade pip
RUN pip install Cython==0.29.28

### Python libs
# todo: can we move it to requirements.txt?
#RUN pip install torch==1.12.0 torchvision==0.13.0 --extra-index-url https://download.pytorch.org/whl/${TORCH_POSTFIX}

COPY ci/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /oml
WORKDIR /oml/
RUN pip install -e .
