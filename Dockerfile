# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/pytorch:22.08-py3

WORKDIR /app

COPY . .

# RUN pip3 install -r requirements.txt

CMD [ "python3", "main.py"]