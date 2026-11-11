# Общие функции: московское время, пути к файлам, маска и геометрия предобработки.
# Зависимости: только opencv и numpy (поэтому модуль работает и вне Docker).
import hashlib
import math
import os
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np

# Москва = UTC+3 круглый год (в России нет перехода на летнее время)
MSK = timezone(timedelta(hours=3), "MSK")


def now_msk():
    return datetime.now(MSK)


def frame_dir(root, ts):
    """<root>/<год>/<месяц>/<день>/<час>"""
    return os.path.join(root, f"{ts:%Y}", f"{ts:%m}", f"{ts:%d}", f"{ts:%H}")


def frame_stem(ts):
    """<час>_<минута>"""
    return f"{ts:%H}_{ts:%M}"


def file_sha1(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Не удалось прочитать маску: {path}")
    return (mask > 127).astype(np.uint8) * 255


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
