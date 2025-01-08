FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

LABEL maintainer="bodi.vencel04@gmail.com"
LABEL description="Docker image for BirdCLEF training"

WORKDIR /chipchirip

RUN DEBIAN_FRONTEND=noninteractive apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -qy \
    python3-pip \
    build-essential \
    autoconf \
    automake \
    sudo \
    vim \
    nano \
    git \
    curl \
    wget \
    tmux \
    openssh-server && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

EXPOSE 22

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

COPY . /chipchirip

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

CMD ["/usr/sbin/sshd", "-D"]