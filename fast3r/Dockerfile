FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV GRADIO_SERVER_NAME 0.0.0.0
ARG DEBIAN_FRONTEND=noninteractive

ARG FORCE_CUDA=1
ENV FORCE_CUDA=${FORCE_CUDA}

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"

RUN apt-get update -y && apt-get install -y git wget
RUN apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6

WORKDIR /
RUN git clone --depth 1 https://github.com/facebookresearch/fast3r
RUN git clone --depth 1 --branch stable https://github.com/facebookresearch/pytorch3d.git

WORKDIR /pytorch3d
RUN pip install -e .

WORKDIR /fast3r
RUN pip install -r requirements.txt
RUN pip install -e .

ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 7860

CMD ["python", "fast3r/viz/demo.py"]