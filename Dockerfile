FROM python:3.7-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV LANGUAGE en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

WORKDIR /src

COPY poetry.lock .
COPY pyproject.toml .
COPY Makefile .

RUN make init

COPY . .
CMD ["make", "run"]
