FROM ubuntu:18.04

# Setup
RUN apt update && apt -y upgrade

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y  software-properties-common r-base libv8-dev libcurl4-openssl-dev libxml2-dev python3.6 python3-pip

# Installing dependencies
RUN pip3 install numpy sklearn minepy pandas numba deap pathos pyswarm sympy shap mlxtend cdt torch

WORKDIR /adan
COPY . .

# Making dowhy globally accessible
RUN ln -s /adan/dowhy/dowhy /usr/lib/python3.6/dowhy
# Install R dependencies
RUN Rscript install.r

# Environment variables
ENV DATASETS_DIR=/adan/datatests
