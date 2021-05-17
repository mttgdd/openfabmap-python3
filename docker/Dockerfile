# os
FROM ubuntu:20.04

# apt
RUN apt-get -y update \
	&& apt-get install -y software-properties-common \
	&& apt-get -y update

# C/C++
RUN apt-get install -y git python3-pip build-essential cmake gcc

# python
RUN echo 'alias python=python3' >> /root/.bashrc \
	&& echo 'alias pip=pip3' >> /root/.bashrc

# this repo
RUN mkdir -p /root/code/openfabmap-python3
WORKDIR /root/code/openfabmap-python3
RUN git clone https://github.com/mttgdd/openfabmap-python3.git src
RUN mkdir build \
	&& cd build \
	&& cmake ../src -DCMAKE_PYTHON_EXECUTABLE=python3 \
	&& make -j5

# entry point at a working dir
ENTRYPOINT ["/bin/bash"]