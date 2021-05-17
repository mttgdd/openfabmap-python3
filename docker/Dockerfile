# os
FROM ubuntu:20.04

# apt
RUN apt-get -y update \
	&& apt-get install -y software-properties-common \
	&& apt-get -y update \
	&& apt-get install -y git python3-pip

# python
RUN echo 'alias python=python3' >> /root/.bashrc \
	&& echo 'alias pip=pip3' >> /root/.bashrc

# pip
RUN git clone https://github.com/mttgdd/openfabmap-python3.git

# entry point at a working dir
ENTRYPOINT ["/bin/bash"]