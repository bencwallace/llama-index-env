FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update --fix-missing && \
	apt-get install -y python3-pip && \
	ln -s /usr/bin/python3 /usr/bin/python

RUN pip install 'poetry==1.7.1'

WORKDIR /workspace
