FROM nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04

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
        python3-gi \
        python3-dev \
        build-essential \
        pkg-config \
        libavcodec-dev \
        libavformat-dev \
        libavdevice-dev \
        libavfilter-dev \
        libavutil-dev \
        libswscale-dev \
        libswresample-dev \
        gir1.2-gstreamer-1.0 \
        gir1.2-gst-plugins-base-1.0 \
        gir1.2-gst-plugins-bad-1.0 \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-nice \
        unzip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv --system-site-packages /opt/feathertalk

ENV VIRTUAL_ENV=/opt/feathertalk \
    PATH="/opt/feathertalk/bin:${PATH}"

RUN python -m pip install --upgrade pip setuptools wheel \
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
    && pip install --no-binary av -r requirements.txt

# Build PyAV against the system FFmpeg rather than wheels that omit NVENC.
RUN python -c "import av; assert 'h264_nvenc' in av.codecs_available, 'FFmpeg/PyAV lacks NVENC'" \
    && gst-inspect-1.0 webrtcbin >/dev/null \
    && gst-inspect-1.0 opusenc >/dev/null

COPY . .

ENV PYTHONFAULTHANDLER=1

RUN mkdir -p audio checkpoints data outputs

EXPOSE 8000
EXPOSE 40000-40100/udp

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/ >/dev/null || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
