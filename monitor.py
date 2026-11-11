#!/usr/bin/env python3
"""Раз в минуту: снимок с камеры -> подсчёт машин -> файлы в ./data.

Для каждого кадра в папке data/<год>/<месяц>/<день>/<час>/ создаются:
  <ЧЧ>_<ММ>_camera.jpeg       исходный снимок с камеры (как получен)
  <ЧЧ>_<ММ>_result.jpeg       визуализация (повёрнутый кадр с рамками и числом машин)
  <ЧЧ>_<ММ>_labels.txt        разметка YOLO (рамки засчитанных машин, для CVAT/обучения)
  <ЧЧ>_<ММ>_detections.json   все найденные объекты с уверенностью, параметры, число машин
  <ЧЧ>_<ММ>_prep.jpeg         повёрнутый кадр (только если SAVE_PREP=1)
  count.csv                   строки «time,count» (общий файл часа)
Время везде московское.
"""
import csv
import json
import logging
import math
import os
import sys
import time

import cv2
import numpy as np

from camera import grab_snapshot
from common import (Geometry, MSK, file_sha1, frame_dir, frame_stem, load_mask, now_msk)

log = logging.getLogger("monitor")

VEHICLE_CLASSES = [2, 5, 7]  # на случай обычной COCO-модели: легковой, автобус, грузовик
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def env(name, default=None, required=False):
    v = os.environ.get(name, default)
    if required and not v:
        sys.exit(f"Не задана переменная окружения {name}")
    return v


class Config:
    def __init__(self):
        self.cam_host = env("CAM_HOST", required=True)
        self.cam_port = int(env("CAM_PORT", "9000"))
        self.cam_user = env("CAM_USER", required=True)
        self.cam_pass = env("CAM_PASS", required=True)
        self.data_dir = env("DATA_DIR", "/data")
        self.model_path = env("MODEL_PATH", "/models/best.pt")
        self.mask_path = env("MASK_PATH", "/app/mask.png")
        self.conf = float(env("CONF", "0.3"))          # порог, при котором машина засчитывается
        self.conf_min = float(env("CONF_MIN", "0.1"))  # ниже не сохраняем даже как «кандидата»
        self.imgsz = int(env("IMGSZ", "1920"))
        self.nms_iou = float(env("NMS_IOU", "0.7"))
        self.device = env("DEVICE", "cpu")
        self.interval = int(env("INTERVAL_SEC", "60"))
        self.vis_width = int(env("VIS_WIDTH", "1600"))
        self.save_prep = env("SAVE_PREP", "0") == "1"
        self.run_once = env("RUN_ONCE", "0") == "1"
        self.uncertain_max = float(env("UNCERTAIN_MAX", "0.6"))


# ------------------------------- Запись файлов -----------------------------
def write_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def write_image(path, img, quality):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Не удалось закодировать {path}")
    write_atomic(path, buf.tobytes())


def append_count(path, ts, count):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "count"])
        w.writerow([ts.isoformat(timespec="seconds"), count])


def yolo_lines(boxes, w, h):
    return "".join(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                   f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n" for x1, y1, x2, y2 in boxes)


# ------------------------------- Визуализация ------------------------------
def make_visual(prep, mask, boxes, confs, in_area, counted, count, ts, width):
    out = prep.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 128, 0), 4)
    for (x1, y1, x2, y2), a, c in zip(boxes, in_area, counted):
        if not a:
            color, th = (0, 0, 255), 2        # красная: вне области
        elif c:
            color, th = (0, 255, 0), 3        # зелёная: засчитана
        else:
            color, th = (0, 215, 255), 2      # жёлтая: в области, но уверенность ниже порога
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, th)
    k = width / out.shape[1]
    out = cv2.resize(out, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    text = f"{ts:%Y-%m-%d %H:%M} МСК   Машин: {count}"
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(FONT_PATH, 34)
        pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        box = d.textbbox((10, 6), text, font=font)
        d.rectangle((0, 0, box[2] + 10, box[3] + 8), fill=(0, 0, 0))
        d.text((10, 6), text, font=font, fill=(0, 255, 0))
        out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.rectangle(out, (0, 0), (420, 50), (0, 0, 0), -1)
        cv2.putText(out, f"{ts:%H:%M} cars: {count}", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
    return out


# ------------------------------- Один тик ----------------------------------
def process_tick(cfg, model, classes, geo, mask_sha1):
    ts = now_msk()
    jpg = grab_snapshot(cfg.cam_host, cfg.cam_port, cfg.cam_user, cfg.cam_pass)
    if jpg is None:
        log.error("%s: снимок с камеры не получен, пропускаю", f"{ts:%H:%M}")
        return False
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        log.error("снимок получен, но не декодируется как JPEG")
        return False

    folder = frame_dir(cfg.data_dir, ts)
    os.makedirs(folder, exist_ok=True)
    stem = os.path.join(folder, frame_stem(ts))
    write_atomic(stem + "_camera.jpeg", jpg)  # сначала сохраняем сырой снимок: он не должен пропасть

    try:
        prep = geo.apply(img)
    except ValueError as e:
        log.error("%s", e)
        return False

    res = model.predict(prep, imgsz=cfg.imgsz, conf=cfg.conf_min, iou=cfg.nms_iou,
                        classes=classes, device=cfg.device, max_det=1000, verbose=False)[0]
    if res.boxes is not None and len(res.boxes):
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
    else:
        boxes, confs = np.zeros((0, 4)), np.zeros(0)

    h, w = prep.shape[:2]
    cx = np.clip(((boxes[:, 0] + boxes[:, 2]) / 2).astype(int), 0, w - 1)
    cy = np.clip(((boxes[:, 1] + boxes[:, 3]) / 2).astype(int), 0, h - 1)
    in_area = geo.mask[cy, cx] > 0 if len(boxes) else np.zeros(0, bool)
    counted = in_area & (confs >= cfg.conf)
    count = int(counted.sum())
    uncertain = int((in_area & (confs >= cfg.conf_min) & (confs < cfg.uncertain_max)).sum())

    write_atomic(stem + "_labels.txt", yolo_lines(boxes[counted], w, h).encode())
    meta = {
        "time": ts.isoformat(timespec="seconds"),
        "count": count,
        "uncertain": uncertain,
        "conf_threshold": cfg.conf, "conf_min": cfg.conf_min, "imgsz": cfg.imgsz,
        "model": os.path.basename(cfg.model_path),
        "camera_size": [img.shape[1], img.shape[0]],
        "prep": {"angle": round(geo.angle, 3), "size": [w, h], "mask_sha1": mask_sha1},
        "detections": [{"box": [round(float(v), 1) for v in b], "conf": round(float(c), 3),
                        "in_area": bool(a)} for b, c, a in zip(boxes, confs, in_area)],
    }
    write_atomic(stem + "_detections.json", json.dumps(meta, ensure_ascii=False).encode())
    write_image(stem + "_result.jpeg",
                make_visual(prep, geo.mask, boxes, confs, in_area, counted, count, ts, cfg.vis_width), 80)
    if cfg.save_prep:
        write_image(stem + "_prep.jpeg", prep, 95)

    append_count(os.path.join(folder, "count.csv"), ts, count)   # строка в csv — в самом конце
    open(os.path.join(cfg.data_dir, ".heartbeat"), "w").close()  # для healthcheck Docker
    log.info("%s: машин %d (неуверенных %d, всего рамок %d)", f"{ts:%H:%M}", count, uncertain, len(boxes))
    return True


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.Formatter.converter = lambda *a: now_msk().timetuple()
    cfg = Config()
    os.makedirs(cfg.data_dir, exist_ok=True)

    mask = load_mask(cfg.mask_path)
    geo = Geometry(mask)
    mask_sha1 = file_sha1(cfg.mask_path)
    log.info("Поворот %.1f°, подготовленный кадр %dx%d, ожидаемый кадр камеры %dx%d",
             geo.angle, geo.size[0], geo.size[1], mask.shape[1], mask.shape[0])

    from ultralytics import YOLO
    model = YOLO(cfg.model_path)
    classes = None if len(model.names) < 10 else VEHICLE_CLASSES
    log.info("Модель %s загружена, порог %.2f, интервал %d с", cfg.model_path, cfg.conf, cfg.interval)

    while True:
        t0 = time.time()
        try:
            process_tick(cfg, model, classes, geo, mask_sha1)
        except Exception:
            log.exception("Ошибка обработки кадра")
        elapsed = time.time() - t0
        if cfg.run_once:
            break
        if elapsed > cfg.interval:
            log.warning("Обработка заняла %.0f с — больше интервала, часть минут будет пропущена", elapsed)
        nxt = (math.floor(time.time() / cfg.interval) + 1) * cfg.interval  # ровно на границу минуты
        time.sleep(max(0.0, nxt - time.time()))


if __name__ == "__main__":
    main()
