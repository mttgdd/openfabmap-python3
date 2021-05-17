# os
FROM ubuntu:16.04

# apt
RUN apt-get -y update \
	&& apt-get install -y software-properties-common \
	&& apt-get -y update \
	&& apt-get -y install wget

# C/C++
RUN apt-get install -y git cmake gcc

# python
RUN add-apt-repository ppa:deadsnakes/ppa \
	&& apt-get -y update \
	&& apt-get -y install python3.6 \
	&& wget https://bootstrap.pypa.io/get-pip.py \
	&& python3.6 get-pip.py \
	&& rm /usr/local/bin/pip3 \
	&& ln -s /usr/bin/python3.6 /usr/local/bin/python3 \
	&& ln -s /usr/local/bin/pip /usr/local/bin/pip3 \
	&& apt-get install -y python3.6-dev python3-pip
RUN echo 'alias python=python3.6' >> /root/.bashrc \
	&& echo 'alias pip=pip3.6' >> /root/.bashrc

# C++ deps
RUN add-apt-repository --remove ppa:xqms/opencv-nonfree \
	&& add-apt-repository --yes ppa:jeff250/opencv \
	&& apt-get -y update \
	&& apt-get install -y libopencv-dev \
	&& apt-get install -y libopencv-nonfree-dev

# numpy
RUN pip install numpy==1.17.0

# this repo
RUN mkdir -p /root/code/openfabmap-python3
WORKDIR /root/code/openfabmap-python3
RUN git clone --recurse-submodules https://github.com/mttgdd/openfabmap-python3.git src
RUN mkdir build \
	&& cd build \
	&& cmake ../src -DPYTHON_EXECUTABLE=/usr/bin/python3.6 \
	&& make -j5

# entry point at a working dir
ENTRYPOINT ["/bin/bash"]