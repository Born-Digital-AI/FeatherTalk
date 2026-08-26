FROM nvidia/cuda:13.0.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        python3 \
        python3-pip \
        python3-venv \
        unzip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && python -m pip install --upgrade pip setuptools wheel \
    && pip install gdown

RUN mkdir -p assets/featherhubert \
    && gdown 1-gSAp_BlQ7xPBDQRCf9cjaYjQT03IOxI -O assets/feathertalk_assets.zip \
    && unzip -q assets/feathertalk_assets.zip -d assets/featherhubert \
    && if [ -d assets/featherhubert/kanghui_training_video_featherhubert_188_latest ]; then \
        cp -a assets/featherhubert/kanghui_training_video_featherhubert_188_latest/. assets/featherhubert/; \
        rm -rf assets/featherhubert/kanghui_training_video_featherhubert_188_latest; \
    fi \
    && rm -f assets/feathertalk_assets.zip

COPY requirements.txt .

RUN pip install \
        torch==2.11.0 \
        torchvision==0.26.0 \
        torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu130 \
    && pip install -r requirements.txt

COPY . .

RUN mkdir -p audio checkpoints data outputs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/ >/dev/null || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
