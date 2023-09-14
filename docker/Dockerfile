FROM  nvidia/cudagl:11.4.1-devel-ubuntu20.04
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    wget \
    curl \
    git \
    vim \
    python3-pip \
    python3.8-venv \
    lsb-release \
    gnupg \
    avahi-autoipd \
    avahi-daemon \
    avahi-utils \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libxrandr2 \
    libxcursor1 \
    libglfw3 \
    libglew2.1


## Install DaXBench's dependencies.
RUN pip install --upgrade pip && \
    pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip install git+https://github.com/fogleman/sdf.git

RUN apt update && \
    apt install -y freeglut3-dev

# Copy the helper scripts to the working directory.
COPY ./docker/scripts/. /root/scripts/
RUN chmod -R +x /root/scripts/*

# # Force color prompt in terminal.
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

ENTRYPOINT []

CMD ["/bin/bash"]