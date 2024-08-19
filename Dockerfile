# Build Stage
FROM debian:bookworm-slim AS builder

RUN apt-get update && \
    apt-get install -y curl git python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && \
    mkdir -p /sam2-playground

WORKDIR /sam2-playground

COPY requirements.txt .

RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt


# Runtime Stage
FROM debian:bookworm-slim AS runtime

RUN apt-get update && \
    apt-get install -y curl ffmpeg python3 && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /sam2-playground

COPY . .
COPY --from=builder /sam2-playground/venv /sam2-playground/venv

ENV PATH="/sam2-playground/venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/sam2-playground/venv/lib64/python3.11/site-packages/nvidia/cublas/lib:/sam2-playground/venv/lib64/python3.11/site-packages/nvidia/cudnn/lib"

ENTRYPOINT ["python", "app.py"]
