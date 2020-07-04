# base layer
FROM ubuntu:latest

# package dependencies
RUN apt-get update &&\
    apt-get install -y python3-dev &&\
    apt-get install -y python3-pip

# python libraries
RUN pip3 install -r /src/requirements.txt

# source code
ADD . /src
WORKDIR /src

FROM python:3.6-onbuild

# network interfaces
EXPOSE 8080

# expose main app
CMD ["main:app"]
