# Подсчёт машин на парковке для Google Colab.
#
# Перед запуском (в первой ячейке Colab):
#   !pip install -q ultralytics
# Загрузите в рабочую папку Colab: mask.png и папку photos/ с кадрами (jpg/png).
# Если рядом лежит дообученная модель best.pt — она будет использована автоматически,
# иначе скачается обычная yolo11x.pt.
#
# Запуск: !python count_cars.py   (или вставьте код в ячейку целиком)
# Перед нейросетью кадр поворачивается так, чтобы самая длинная сторона области стала
# горизонтальной, обрезается по области, а всё вне области закрашивается чёрным.
#
# Результат для каждого photos/NAME.jpg:
#   out/NAME.jpg  — подготовленный кадр с рамками (зелёные — в области, красные — вне её)
#   out/NAME.json — число машин
#   out/prep/NAME.jpg   — подготовленный (повёрнутый и обрезанный) кадр без рамок
#   out/labels/NAME.txt — предразметка YOLO в координатах подготовленного кадра
# Общая таблица: out/summary.csv

import csv
import glob
import json
import math
import os
import time
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ------------------------------- Настройки ---------------------------------
PHOTOS_DIR = "./photos"
OUT_DIR = "./out"
MASK_PATH = "./mask.png"
MODEL_PATH = "./best.pt" if os.path.exists("./best.pt") else "yolo11x.pt"

CONF = 0.25          # порог уверенности
TILE = 1280          # размер тайла, px (0 — весь кадр одним куском)
IMGSZ = 1280         # размер, до которого тайл растягивается для нейросети
OVERLAP = 0.25       # перекрытие тайлов
NMS_IOU = 0.7        # порог IoU внутри нейросети
IOS = 0.7            # порог подавления вложенных боксов при слиянии тайлов
AUGMENT = False      # TTA: медленнее, иногда находит больше машин
SAVE_LABELS = True   # сохранять предразметку YOLO в out/labels
MARGIN = 40          # px: запас вокруг области при обрезке и закрашивании
ROTATE_DEG = None    # поворот, градусы (OpenCV: «+» против часовой); None — авто
PREP_SCALE = 1.0     # масштаб подготовленного кадра (0.7 — уменьшить на 30%)

# Для дообученной best.pt (один класс «car») используются эти настройки
CUSTOM_TILE = 0
CUSTOM_IMGSZ = 1920

DEVICE = 0 if torch.cuda.is_available() else "cpu"
EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")
# Классы COCO: 2 — легковой, 5 — автобус, 7 — грузовик
VEHICLE_CLASSES = [2, 5, 7]
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


# ------------------------- Предобработка (поворот) --------------------------
class Geometry:
    def __init__(self, mask, angle=None, margin=40, scale=1.0):
        """mask — маска области в координатах исходного кадра.
        angle   — поворот в градусах (как в OpenCV: «+» против часовой). None — авто:
                  самая длинная сторона области становится горизонтальной.
        margin  — запас вокруг области, px.  scale — масштаб результата (1.0 — без изменений)."""
        mask = (mask > 127).astype(np.uint8) * 255
        cnt = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                  key=cv2.contourArea)
        if angle is None:
            angle = self._auto_angle(cnt)
        self.angle = float(angle)
        self.src_shape = mask.shape

        m = cv2.moments(cnt)
        center = (m["m10"] / m["m00"], m["m01"] / m["m00"])
        M = cv2.getRotationMatrix2D(center, self.angle, scale)
        pts = cv2.transform(cnt.astype(np.float32), M).reshape(-1, 2)
        xmin, ymin = pts.min(axis=0) - margin
        xmax, ymax = pts.max(axis=0) + margin
        M[0, 2] -= xmin
        M[1, 2] -= ymin
        self.M = M
        self.size = (int(math.ceil(xmax - xmin)), int(math.ceil(ymax - ymin)))  # (w, h)

        self.mask = cv2.warpAffine(mask, M, self.size, flags=cv2.INTER_NEAREST)
        self.mask = (self.mask > 127).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
        self.wide = cv2.dilate(self.mask, k)

    @staticmethod
    def _auto_angle(cnt):
        ap = cv2.approxPolyDP(cnt, 0.004 * cv2.arcLength(cnt, True), True)[:, 0, :]
        best, theta = 0.0, 0.0
        for i in range(len(ap)):
            p, q = ap[i], ap[(i + 1) % len(ap)]
            length = math.hypot(*(q - p))
            if length > best:
                best = length
                theta = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))
        r = (180 - theta) % 180        # поворот по часовой стрелке до горизонтали
        if r > 90:
            r -= 180                   # выбираем поворот поменьше
        return -r                      # OpenCV: «+» — против часовой

    def apply(self, img):
        """Повёрнутый, обрезанный кадр; всё вне области (с запасом) — чёрное."""
        if img.shape[:2] != self.src_shape:
            raise ValueError(f"Размер кадра {img.shape[1]}x{img.shape[0]} не совпадает "
                             f"с маской {self.src_shape[1]}x{self.src_shape[0]}")
        out = cv2.warpAffine(img, self.M, self.size, flags=cv2.INTER_LINEAR, borderValue=0)
        out[self.wide == 0] = 0
        return out


# ------------------------------- Маска -------------------------------------
def load_mask(path, shape):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Не найдена маска: {path}. Загрузите mask.png в рабочую папку.")
    if mask.shape != shape[:2]:
        print(f"  Размер маски {mask.shape[::-1]} не совпадает с кадром "
              f"{shape[:2][::-1]} — масштабирую")
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8) * 255


# ------------------------------- Детекция ----------------------------------
def tile_coords(x0, y0, x1, y1, tile, overlap):
    """Окна tile×tile, покрывающие прямоугольник с перекрытием."""
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(x0, max(x0 + 1, x1 - tile + 1), step))
    ys = list(range(y0, max(y0 + 1, y1 - tile + 1), step))
    if not xs or xs[-1] + tile < x1:
        xs.append(max(x0, x1 - tile))
    if not ys or ys[-1] + tile < y1:
        ys.append(max(y0, y1 - tile))
    return [(x, y, min(x + tile, x1), min(y + tile, y1)) for y in ys for x in xs]


def merge_boxes(boxes, scores, iou_thr=0.5, ios_thr=IOS):
    """NMS по IoU + подавление «обрезанных» боксов (пересечение / меньшая площадь)."""
    if len(boxes) == 0:
        return []
    order = np.argsort(-scores)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    for i in order:
        ok = True
        for j in keep:
            iw = min(boxes[i, 2], boxes[j, 2]) - max(boxes[i, 0], boxes[j, 0])
            ih = min(boxes[i, 3], boxes[j, 3]) - max(boxes[i, 1], boxes[j, 1])
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            if inter / (area[i] + area[j] - inter) > iou_thr or \
               inter / min(area[i], area[j]) > ios_thr:
                ok = False
                break
        if ok:
            keep.append(i)
    return keep


def detect_cars(model, img, mask, classes):
    """Возвращает (boxes[N,4], scores[N]) всех машин рядом с областью."""
    ys, xs = np.where(mask > 0)
    pad = 40  # запас, чтобы не терять машины у самой границы
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1, y1 = min(img.shape[1], xs.max() + 1 + pad), min(img.shape[0], ys.max() + 1 + pad)

    windows = ([(x0, y0, x1, y1)] if TILE <= 0
               else tile_coords(x0, y0, x1, y1, TILE, OVERLAP))
    all_boxes, all_scores = [], []
    for (tx0, ty0, tx1, ty1) in windows:
        res = model.predict(img[ty0:ty1, tx0:tx1], imgsz=IMGSZ, conf=CONF, iou=NMS_IOU,
                            classes=classes, device=DEVICE, augment=AUGMENT,
                            verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        b = res.boxes.xyxy.cpu().numpy().copy()
        b[:, [0, 2]] += tx0
        b[:, [1, 3]] += ty0
        all_boxes.append(b)
        all_scores.append(res.boxes.conf.cpu().numpy())

    if not all_boxes:
        return np.zeros((0, 4)), np.zeros(0)
    boxes, scores = np.concatenate(all_boxes), np.concatenate(all_scores)
    keep = merge_boxes(boxes, scores)
    return boxes[keep], scores[keep]


def inside_mask(boxes, mask):
    """True для боксов, центр которых лежит внутри маски."""
    h, w = mask.shape
    flags = []
    for x1, y1, x2, y2 in boxes:
        cx = int(np.clip((x1 + x2) / 2, 0, w - 1))
        cy = int(np.clip((y1 + y2) / 2, 0, h - 1))
        flags.append(mask[cy, cx] > 0)
    return np.array(flags, dtype=bool)


# ------------------------------- Вывод -------------------------------------
def put_text_ru(img, text):
    """Пишет подпись кириллицей (через PIL); если шрифта нет — латиницей через OpenCV."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(FONT_PATH, 64)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        box = d.textbbox((15, 8), text, font=font)
        d.rectangle((0, 0, box[2] + 15, box[3] + 10), fill=(0, 0, 0))
        d.text((15, 8), text, font=font, fill=(0, 255, 0))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.rectangle(img, (0, 0), (520, 90), (0, 0, 0), -1)
        cv2.putText(img, f"Cars: {text.split(': ')[-1]}", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        return img


def draw(img, mask, boxes, flags, count):
    out = img.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 128, 0), 4)
    for (x1, y1, x2, y2), ok in zip(boxes, flags):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 255, 0) if ok else (0, 0, 255), 2)
    return put_text_ru(out, f"Машин в области: {count}")


def save_yolo_labels(path, img, boxes):
    h, w = img.shape[:2]
    with open(path, "w") as f:
        for x1, y1, x2, y2 in boxes:
            f.write(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                    f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")


# ------------------------------- Основное ----------------------------------
def main():
    files = sorted(f for ext in EXTENSIONS
                   for f in glob.glob(os.path.join(PHOTOS_DIR, ext)))
    if not files:
        print(f"В папке {PHOTOS_DIR} нет изображений (jpg/jpeg/png).")
        return
    os.makedirs(os.path.join(OUT_DIR, "prep"), exist_ok=True)
    if SAVE_LABELS:
        os.makedirs(os.path.join(OUT_DIR, "labels"), exist_ok=True)

    print(f"Устройство: {'GPU' if DEVICE != 'cpu' else 'CPU'}; модель: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    classes = None if len(model.names) < 10 else VEHICLE_CLASSES  # дообученная: 1 класс

    global TILE, IMGSZ
    if classes is None:  # дообученная модель
        TILE, IMGSZ = CUSTOM_TILE, CUSTOM_IMGSZ

    geo, summary = None, []
    for n, path in enumerate(files, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[{n}/{len(files)}] {name}: не удалось прочитать файл — пропускаю")
            continue
        if geo is None:
            geo = Geometry(load_mask(MASK_PATH, img.shape), ROTATE_DEG, MARGIN, PREP_SCALE)
            print(f"Поворот: {geo.angle:.1f}°, размер подготовленного кадра: "
                  f"{geo.size[0]}x{geo.size[1]} (было {img.shape[1]}x{img.shape[0]})")
        try:
            prep = geo.apply(img)
        except ValueError as exc:
            print(f"[{n}/{len(files)}] {name}: {exc} — пропускаю")
            continue
        mask = geo.mask

        t0 = time.time()
        boxes, _ = detect_cars(model, prep, mask, classes)
        flags = inside_mask(boxes, mask) if len(boxes) else np.zeros(0, bool)
        count = int(flags.sum())
        sec = round(time.time() - t0, 1)

        cv2.imwrite(os.path.join(OUT_DIR, name + ".jpg"),
                    draw(prep, mask, boxes, flags, count), [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(os.path.join(OUT_DIR, "prep", name + ".jpg"), prep,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(OUT_DIR, name + ".json"), "w", encoding="utf-8") as f:
            json.dump({"file": os.path.basename(path),
                       "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                       "cars_in_area": count,
                       "cars_detected_total": int(len(boxes)),
                       "processing_seconds": sec}, f, ensure_ascii=False, indent=2)
        if SAVE_LABELS:
            save_yolo_labels(os.path.join(OUT_DIR, "labels", name + ".txt"), prep, boxes[flags])

        summary.append((os.path.basename(path), count, len(boxes)))
        print(f"[{n}/{len(files)}] {name}: машин в области {count} "
              f"(всего найдено {len(boxes)}), {sec} с")

    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["файл", "машин_в_области", "найдено_всего"])
        w.writerows(summary)
    print(f"Готово. Результаты в {OUT_DIR}/")


if __name__ == "__main__":
    main()