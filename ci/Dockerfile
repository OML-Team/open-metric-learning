ARG RUNTIME=cpu

FROM nvidia/cuda:11.1.1-devel-ubuntu18.04 AS gpu_base

ENV EXTRA_INDEX=''

FROM ubuntu:18.04 AS cpu_base

ENV EXTRA_INDEX="--extra-index-url https://download.pytorch.org/whl/cpu"


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
RUN pip3 install --upgrade pip wheel setuptools
RUN pip install Cython==0.29.28

### Python libs
COPY ci/requirements.txt requirements.txt
COPY ci/requirements_tests.txt requirements_tests.txt
COPY ci/requirements_optional.txt requirements_optional.txt
RUN pip install --no-cache-dir -r requirements.txt ${EXTRA_INDEX}
RUN pip install --no-cache-dir -r requirements_optional.txt
RUN pip install --no-cache-dir -r requirements_tests.txt

WORKDIR /oml-lib
COPY . /oml-lib
RUN pip install .

CMD ["sh", "-c", "while true; do sleep 86400; done"]
