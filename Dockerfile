#---
# name: cascade-pipeline
# group: app
# depends: [torchaudio, transformers, faster-whisper, whisper_s2t]
# requires: '>=34.1.0'
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

VOLUME /model-cache

# Set both just in case old version of transformers is used
ENV TRANSFORMERS_CACHE="/model-cache/models"
ENV HF_HOME="/model-cache/hf"

RUN apt-get update && \
    apt install -y libasound2-dev alsa-utils && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean


RUN pip3 install --no-cache-dir evaluate jiwer sacrebleu librosa soundfile uroman \
            more_itertools pypinyin pyalsaaudio pyserial

# wtpsplit depends on onnxruntime, but pip cannot detect that it is satisfied by onnxruntime-gpu
# which is installed. So install all the other required dependencies manually
RUN pip3 install --no-cache-dir adapters>=1.0.1 mosestokenizer skops cached_property
RUN pip3 install --no-cache-dir --no-dependencies wtpsplit

WORKDIR /opt

COPY . /opt/pipeline
WORKDIR /opt/pipeline

CMD bash
