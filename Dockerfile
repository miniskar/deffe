FROM ubuntu:22.04
LABEL maintainer="Narasinga Rao Miniskar"


COPY apt.conf /etc/apt/apt.conf
ENV http_proxy "http://proxy.ftpn.ornl.gov:3128"
ENV https_proxy "http://proxy.ftpn.ornl.gov:3128"

# Install tools for development.
RUN apt-get update --fix-missing

RUN apt-get install -y \
  python3 \
  python3-pip \
  git \
  tmux \
  vim \
  cmake \
  wget \
  ack-grep \
  scons \
  build-essential \
  python3-dev \
  m4 \
  libprotobuf-dev \
  python3-protobuf \
  protobuf-compiler \
  libgoogle-perftools-dev \
  graphviz 

  RUN pip3 install --upgrade pip

# Environment variables for gem5-aladdin
ENV SHELL /bin/bash
RUN mkdir -p /home

COPY . /home/deffe/
WORKDIR /home/deffe
RUN pip3 install -e /home/deffe
WORKDIR /home/deffe/example
RUN run_deffe -h 
RUN git clone https://gem5.googlesource.com/public/gem5 /home/gem5
RUN ls -al
RUN pwd
WORKDIR /home/gem5
ENV GEM5_DIR /home/gem5
RUN scons -j8 build/RISCV/gem5.opt 
RUN ls -al
WORKDIR /home/deffe
