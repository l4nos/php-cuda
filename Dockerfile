# Reference build environment: Ubuntu 24.04 + PHP + CUDA 12.x + cuDNN 9
#
# Build:  docker build -t php-cuda .
# Run (GPU required):  docker run --gpus all --rm php-cuda php -m
# Tests:               docker run --gpus all --rm php-cuda bash -c 'cd /src/plugin && php run-tests.php -q -d extension=$PWD/modules/cuda.so ../tests/'

FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        php-cli php-dev \
        build-essential autoconf pkg-config \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY plugin/ /src/plugin/
COPY tests/ /src/tests/

WORKDIR /src/plugin
RUN phpize \
    && ./configure --with-cuda=/usr/local/cuda --with-cudnn=/usr/local/cuda --with-nvrtc=/usr/local/cuda \
    && make -j"$(nproc)" \
    && make install \
    && echo "extension=cuda.so" > "$(php-config --ini-dir)/cuda.ini"

# Smoke check: the extension must load even without a GPU present at build time.
RUN php -m | grep -i cuda

CMD ["php", "-m"]
