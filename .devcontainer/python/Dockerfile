FROM python:3.11-slim

WORKDIR /tmp/work

COPY requirements.txt ${PWD}

RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y git
