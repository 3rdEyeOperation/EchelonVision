FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install RKNN Lite runtime - keep original wheel filename (pip validates wheel names)
COPY rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl /tmp/
RUN pip install --no-cache-dir /tmp/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl \
    && rm /tmp/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# Bundle RKNN runtime native library into the image.
COPY librknnrt.so /usr/lib/librknnrt.so
RUN chmod 644 /usr/lib/librknnrt.so || true

COPY app ./app
COPY 3rdEchelonLogo.svg ./3rdEchelonLogo.svg
COPY convert_model_to_json.py ./convert_model_to_json.py
COPY model_bootstrap.py ./model_bootstrap.py
COPY start.sh ./start.sh

RUN mkdir -p /data/faces
RUN mkdir -p /models
RUN chmod +x /app/start.sh

EXPOSE 3000

CMD ["/app/start.sh"]
