FROM ubuntu:18.04
LABEL maintainer="Narasinga Rao Miniskar"


COPY apt.conf /etc/apt/apt.conf
ENV http_proxy "http://proxy.ftpn.ornl.gov:3128"
ENV https_proxy "http://proxy.ftpn.ornl.gov:3128"

# Install tools for development.
RUN apt-get update 
RUN apt-get install -y \
  python3 \
  python3-pip \
  git \
  tmux \
  vim \
  cmake \
  wget \
  ack-grep \
  graphviz 

# Install a supported version of pyparsing for Xenon.
RUN pip3 install keras \
        tensorflow \
        tensorflow-gpu \
        torch \
        scikit-learn \
        xlsxwriter \
        matplotlib \
        pandas \
        pydot \
        tqdm \
        torchsummary


# Environment variables for gem5-aladdin
ENV SHELL /bin/bash
RUN mkdir -p /workspace
COPY . /workspace/deffe/
WORKDIR /workspace/deffe
ENV DEFFE_DIR /workspace/deffe
RUN ls -al
WORKDIR /workspace/deffe/example
RUN python3 $DEFFE_DIR/framework/run_deffe.py -h 
