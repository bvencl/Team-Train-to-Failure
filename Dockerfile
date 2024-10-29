FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

LABEL maintainer1="bodi.vencel04@gmail.com"
LABEL maintainer2="mitrengamark@edu.bme.hu"
LABEL docker_image_name="BMEVITMAV45"
LABEL description="This container is created to develop a CNN model to solve the BirdCLEF problem"
LABEL com.centurylinklabs.watchtower.enable="true"

WORKDIR /bmevitmav

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
    openssh-server

RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

RUN mkdir /root/.ssh

EXPOSE 22

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

COPY . .

RUN pip3 install -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/"
ENV PYTHONPATH "${PYTHONPATH}:/bmevitmav/"

RUN echo "cd /bmevitmav" >> /root/.bashrc

CMD ["/usr/sbin/sshd", "-D"]