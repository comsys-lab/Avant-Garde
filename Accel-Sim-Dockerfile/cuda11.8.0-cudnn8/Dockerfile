FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# apt-get install할 때 사용자 입력 안 받게 하기
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update
RUN apt-get install -y wget build-essential xutils-dev bison zlib1g-dev flex 
RUN apt-get install -y libglu1-mesa-dev git g++ libssl-dev libxml2-dev libboost-all-dev git g++
RUN apt-get install -y libxml2-dev vim python-setuptools 
RUN apt-get install -y python-dev build-essential python3-pip

RUN pip3 install pyyaml plotly psutil
RUN wget http://developer.download.nvidia.com/compute/cuda/11.0.1/local_installers/cuda_11.0.1_450.36.06_linux.run
RUN sh cuda_11.0.1_450.36.06_linux.run --silent --toolkit
RUN rm cuda_11.0.1_450.36.06_linux.run

# Environmental setting 
ENV CUDA_INSTALL_PATH /usr/local/cuda-11.6
ENV PATH $CUDA_INSTALL_PATH/bin:$PATH
RUN rm /usr/local/cuda \
    && ln -s /usr/local/cuda-11.6 /usr/local/cuda

# Install AccelSim 
WORKDIR /workspace
RUN git clone https://github.com/accel-sim/accel-sim-framework.git \
    && mv accel-sim-framework accel-sim-dev


# Install NVbit tracer 
WORKDIR /workspace/accel-sim-dev
RUN ./util/tracer_nvbit/install_nvbit.sh \
    && cp -r ./nvbit_release ./util/tracer_nvbit \
    && make -C ./util/tracer_nvbit/

# Accel-Sim SASS Frontend and Simulation Engine
RUN pip3 install -r requirements.txt
SHELL ["/bin/bash", "-c"]
RUN source ./gpu-simulator/setup_environment.sh; \
    git checkout dev; \
    make -j -C ./gpu-simulator/

# Create directory to mount
RUN mkdir -p /workspace/results && mkdir -p /workspace/scripts && mkdir -p /workspace/workloads

ARG REF=main
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF
RUN python3 -m pip install --no-cache-dir -e ./transformers[dev-torch,testing,video]

# If set to nothing, will install the latest version
ARG PYTORCH='2.0.0'
ARG TORCH_VISION=''
ARG TORCH_AUDIO=''
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu117'

RUN [ ${#PYTORCH} -gt 0 ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA
RUN [ ${#TORCH_VISION} -gt 0 ] && VERSION='torchvision=='TORCH_VISION'.*' ||  VERSION='torchvision'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA
RUN [ ${#TORCH_AUDIO} -gt 0 ] && VERSION='torchaudio=='TORCH_AUDIO'.*' ||  VERSION='torchaudio'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA

RUN python3 -m pip uninstall -y tensorflow flax

RUN python3 -m pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2.git pytesseract
RUN python3 -m pip install -U "itsdangerous<2.1.0"

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop