ARG RUNTIME=cpu


FROM nvidia/cuda:11.1-devel-ubuntu18.04 AS gpu_base

ENV TORCH_POSTFIX=cu111

### NVIDIA keys trouble
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb


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
RUN pip install torch==1.12.0+${TORCH_POSTFIX} torchvision==0.13.0+${TORCH_POSTFIX} \
    -f https://download.pytorch.org/whl/torch_stable.html

COPY ci/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /oml
WORKDIR /oml/
RUN pip install -e .
