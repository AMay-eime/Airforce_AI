FROM ubuntu:22.04

ENV PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    TZ="Asia/Tokyo" \
    LANG=ja_JP.UTF-8 \
    LANGUAGE=ja_JP:en \
    PATH=$PATH:/opt/conda/bin

RUN apt-get -qq -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install \
        build-essential \
        wget \
        gcc \
        cmake \
        libeigen3-dev \
        nlohmann-json3-dev \
        libboost-dev \
        libnlopt-cxx-dev \
        freeglut3-dev \
        xvfb \
        tzdata \
        locales \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && cp /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

WORKDIR /usr/local/src

ARG ANACONDA3_VERSION=2022.10

RUN wget -q https://repo.continuum.io/archive/Anaconda3-${ANACONDA3_VERSION}-Linux-x86_64.sh -P ./downloads/ && \
    bash ./downloads/Anaconda3-${ANACONDA3_VERSION}-Linux-x86_64.sh -b -p /opt/conda && \
    rm -rf ./downloads

WORKDIR /usr/local/include

COPY root/thirdParty/include/pybind11_json/pybind11_json.hpp pybind11_json/pybind11_json.hpp
ADD https://github.com/Neargye/magic_enum/releases/download/v0.7.3/magic_enum.hpp magic_enum/magic_enum.hpp
ADD https://raw.githubusercontent.com/bshoshany/thread-pool/v3.3.0/BS_thread_pool.hpp thread-pool/thread_pool.hpp

RUN chmod 644 pybind11_json/pybind11_json.hpp magic_enum/magic_enum.hpp thread-pool/thread_pool.hpp

WORKDIR /usr/local/src

COPY root ASRCAISim1
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install ./ASRCAISim1/sample/OriginalModelSample && \
    rm -rf requirements.txt ASRCAISim1 && \
    pip cache purge

COPY root/sample/MinimumEvaluation /work

RUN useradd signate -m -u 1000
RUN chown -R signate:signate /work

WORKDIR /work

USER signate
