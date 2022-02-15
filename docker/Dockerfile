# docker build -t huggingface_asr:v4.16.2 .

#use the pytorch image recommended by NeMo
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install libsndfile1 (linux soundfile package)
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg wget gcc git git-lfs \
&& rm -rf /var/lib/apt/lists/*

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Install pip requirements
ADD requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 -m pip install --no-cache-dir torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

#docker container starts with bash
WORKDIR /vc
RUN ["bash"]