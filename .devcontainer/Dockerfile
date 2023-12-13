FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

# Update default packages
RUN apt-get update

# Get Ubuntu packages
RUN apt-get install -y \
    build-essential \
    curl \
    git \
    python3-dev \
    gcc\
    python3-pip

# Update new packages
RUN apt-get update
RUN apt-get install -y git python3-dev gcc python3-pip stockfish
RUN python3 -m pip install --upgrade pip setuptools wheel 

WORKDIR /app
CMD ["/bin/bash"]

