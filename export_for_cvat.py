#!/usr/bin/env python3
"""Отбирает кадры из ./data и готовит их для доразметки в CVAT.

Результат в папке --out:
  images/ИМЯ.jpg      повёрнутые и обрезанные кадры (те же, на которых работала сеть)
  cvat_images.zip     эти же кадры одним архивом — для создания задачи в CVAT
  cvat_import.zip     предразметка (то, что нашла сеть) в формате «YOLO 1.1» — для импорта в задачу
  frames.csv          список отобранных кадров

Запуск на обычном компьютере (нужны только opencv и numpy):
  python export_for_cvat.py --since 2026-11-12 --until 2026-11-14 -n 40
"""
import argparse
import csv
import glob
import json
import os
import random
import shutil
import sys
import zipfile
from datetime import datetime, timedelta

import cv2
import numpy as np

from common import Geometry, file_sha1, load_mask

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--data", default="./data")
p.add_argument("--mask", default="./mask.png", help="та же маска, что у работающего контейнера")
p.add_argument("--out", help="папка результата (по умолчанию ./cvat_pack_ДАТА_ВРЕМЯ)")
p.add_argument("--since", help="с даты (ГГГГ-ММ-ДД), включительно")
p.add_argument("--until", help="по дату (ГГГГ-ММ-ДД), включительно")
p.add_argument("--hours", help="только часы суток, например 7-22")
p.add_argument("-n", type=int, default=40, help="сколько кадров взять")
p.add_argument("--strategy", choices=["uncertain", "uniform", "random"], default="uncertain",
               help="uncertain — где сеть наименее уверена; uniform — равномерно по времени; random")
p.add_argument("--min-gap", type=int, default=20, help="минимум минут между кадрами (кроме uniform)")
p.add_argument("--include-exported", action="store_true", help="не исключать уже выгружавшиеся кадры")
p.add_argument("--seed", type=int, default=1)
a = p.parse_args()
if not a.out:
    a.out = f"./cvat_pack_{datetime.now():%Y%m%d_%H%M}"

mask = load_mask(a.mask)
mask_sha1 = file_sha1(a.mask)
geo = Geometry(mask)
exported_log = os.path.join(a.data, "exported_frames.txt")
exported = set(open(exported_log).read().split()) if os.path.exists(exported_log) else set()

since = datetime.strptime(a.since, "%Y-%m-%d") if a.since else None
until = datetime.strptime(a.until, "%Y-%m-%d") + timedelta(days=1) if a.until else None
h0, h1 = (map(int, a.hours.split("-")) if a.hours else (0, 23))

cands, bad_mask = [], 0
for jp in glob.glob(os.path.join(a.data, "*", "*", "*", "*", "*_detections.json")):
    parts = os.path.normpath(jp).split(os.sep)
    y, mo, d, h, fn = parts[-5:]
    mi = int(fn.split("_")[1])
    dt = datetime(int(y), int(mo), int(d), int(h), mi)
    if (since and dt < since) or (until and dt >= until) or not h0 <= dt.hour <= h1:
        continue
    name = f"{dt:%Y_%m_%d__%H_%M}"
    if name in exported and not a.include_exported:
        continue
    base = jp[: -len("_detections.json")]
    if not (os.path.exists(base + "_camera.jpeg") and os.path.exists(base + "_labels.txt")):
        continue
    meta = json.load(open(jp, encoding="utf-8"))
    if meta["prep"]["mask_sha1"] != mask_sha1:
        bad_mask += 1   # кадр обработан с другой маской — рамки не совпадут с текущей геометрией
        continue
    cands.append({"dt": dt, "name": name, "base": base, "count": meta["count"],
                  "uncertain": meta.get("uncertain", 0)})
if bad_mask:
    print(f"Пропущено кадров с другой маской: {bad_mask}")
if not cands:
    sys.exit("Подходящих кадров не найдено.")

rng = random.Random(a.seed)
gap = timedelta(minutes=a.min_gap)
if a.strategy == "uniform":
    cands.sort(key=lambda c: c["dt"])
    idx = sorted(set(np.linspace(0, len(cands) - 1, min(a.n, len(cands))).round().astype(int)))
    chosen = [cands[i] for i in idx]
else:
    if a.strategy == "uncertain":
        rng.shuffle(cands)
        cands.sort(key=lambda c: -c["uncertain"])
    else:
        rng.shuffle(cands)
    chosen = []
    for c in cands:
        if len(chosen) >= a.n:
            break
        if all(abs(c["dt"] - s["dt"]) >= gap for s in chosen):
            chosen.append(c)
chosen.sort(key=lambda c: c["dt"])

shutil.rmtree(a.out, ignore_errors=True)
os.makedirs(os.path.join(a.out, "images"))
build = os.path.join(a.out, "_yolo")
os.makedirs(os.path.join(build, "obj_train_data"))
for c in chosen:
    img = cv2.imread(c["base"] + "_camera.jpeg")
    prep = geo.apply(img)  # та же предобработка, что при подсчёте
    for dst in (os.path.join(a.out, "images", c["name"] + ".jpg"),
                os.path.join(build, "obj_train_data", c["name"] + ".jpg")):
        cv2.imwrite(dst, prep, [cv2.IMWRITE_JPEG_QUALITY, 95])
    shutil.copy(c["base"] + "_labels.txt", os.path.join(build, "obj_train_data", c["name"] + ".txt"))

names = [c["name"] for c in chosen]
open(f"{build}/obj.names", "w").write("car\n")
open(f"{build}/obj.data", "w").write("classes = 1\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n")
open(f"{build}/train.txt", "w").write("".join(f"data/obj_train_data/{n}.jpg\n" for n in names))

with zipfile.ZipFile(os.path.join(a.out, "cvat_images.zip"), "w", zipfile.ZIP_DEFLATED) as z:
    for n in names:
        z.write(os.path.join(a.out, "images", n + ".jpg"), n + ".jpg")
with zipfile.ZipFile(os.path.join(a.out, "cvat_import.zip"), "w", zipfile.ZIP_DEFLATED) as z:
    for f in ("obj.names", "obj.data", "train.txt"):
        z.write(os.path.join(build, f), f)
    for n in names:
        for ext in ("jpg", "txt"):
            z.write(os.path.join(build, "obj_train_data", f"{n}.{ext}"), f"obj_train_data/{n}.{ext}")
shutil.rmtree(build)

with open(os.path.join(a.out, "frames.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["name", "time_msk", "count", "uncertain"])
    for c in chosen:
        w.writerow([c["name"], f"{c['dt']:%Y-%m-%d %H:%M}", c["count"], c["uncertain"]])
with open(exported_log, "a") as f:
    f.write("".join(n + "\n" for n in names))

print(f"Готово: {len(chosen)} кадров из {len(cands)} подходящих -> {a.out}/")
print("  cvat_images.zip — кадры для задачи CVAT, cvat_import.zip — предразметка (YOLO 1.1)")
