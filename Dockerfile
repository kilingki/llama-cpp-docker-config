ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

ARG LLAMA_CPP_TAG=b10453
ARG CUDA_ARCH=86
ARG GCC_VERSION=14

ENV DEBIAN_FRONTEND=noninteractive
ENV CC=gcc-${GCC_VERSION}
ENV CXX=g++-${GCC_VERSION}
ENV CUDAHOSTCXX=g++-${GCC_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc-${GCC_VERSION} \
        g++-${GCC_VERSION} \
        build-essential \
        cmake \
        git \
        ca-certificates \
        libssl-dev \
        libcurl4-openssl-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch ${LLAMA_CPP_TAG} https://github.com/ggml-org/llama.cpp.git .

# CUDA driver lib is injected at runtime; link against stubs during the image build.
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}

RUN cmake -B build \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
        -DCMAKE_SHARED_LINKER_FLAGS=-Wl,--allow-shlib-undefined

RUN cmake --build build --config Release -j"$(nproc)"

RUN mkdir -p /opt/llama/bin /opt/llama/lib \
    && cp build/bin/llama-server /opt/llama/bin/ \
    && if [ -f build/bin/llama-bench ]; then cp build/bin/llama-bench /opt/llama/bin/; fi \
    && find build -name "*.so*" -exec cp -P {} /opt/llama/lib \;

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        libgomp1 \
        libcurl4 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/llama/bin/ /usr/local/bin/
COPY --from=build /opt/llama/lib/ /usr/local/lib/
RUN ldconfig

WORKDIR /app
COPY controller/requirements.txt /app/controller/requirements.txt
RUN pip3 install --break-system-packages --no-cache-dir -r /app/controller/requirements.txt

COPY controller/ /app/controller/

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "controller.main:app", "--host", "0.0.0.0", "--port", "8000"]
