FROM python:3.10.9
ENV PYTHONUNBUFFERED 1
RUN mkdir /code

WORKDIR /code

COPY requirements.txt /code/

RUN python -m pip install -r requirements.txt

#RUN apt-get install freeglut3 freeglut3-dev

COPY . /code/