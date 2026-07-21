# INSV->data worker (CPU stage 0). Built amd64 by RunPod's GitHub builder.
FROM python:3.11-slim

# ffmpeg = for .insv stitching + video decode; libgl/glib = OpenCV runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir runpod opencv-python-headless numpy

COPY pipeline/ /app/pipeline/
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
