FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/mpl

RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i \
            -e 's|http://deb.debian.org/debian|http://mirror.yandex.ru/debian|g' \
            -e 's|http://security.debian.org/debian-security|http://mirror.yandex.ru/debian-security|g' \
            /etc/apt/sources.list.d/debian.sources; \
    fi && \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i \
            -e 's|http://deb.debian.org/debian|http://mirror.yandex.ru/debian|g' \
            -e 's|http://security.debian.org/debian-security|http://mirror.yandex.ru/debian-security|g' \
            /etc/apt/sources.list; \
    fi

# libgl1 нужен, потому что ultralytics подтягивает обычный opencv-python;
# fonts-dejavu-core — шрифт с кириллицей для подписи на картинках
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-версия torch (заметно легче обычной)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Модель (/models/best.pt), маска (/app/mask.png) и данные (/data) монтируются снаружи
COPY ConnectionLayer.py camera.py common.py monitor.py ./

CMD ["python", "monitor.py"]
