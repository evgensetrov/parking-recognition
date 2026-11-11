# Шаг 1. Готовит кадры и предразметку для загрузки в CVAT.
# Нужен результат работы count_cars_colab.py: ./out/prep (подготовленные кадры)
# и ./out/labels (предразметка в тех же координатах).
# Результат: cvat_images.zip (кадры для задачи) и cvat_import.zip (предразметка YOLO 1.1)
import glob
import os
import shutil
import zipfile

PREP_DIR = "./out/prep"
LABELS_DIR = "./out/labels"
BUILD = "./cvat_build"

files = sorted(glob.glob(os.path.join(PREP_DIR, "*.jpg")))
if not files:
    raise SystemExit(f"В {PREP_DIR} нет кадров. Сначала запустите count_cars_colab.py")

shutil.rmtree(BUILD, ignore_errors=True)
os.makedirs(f"{BUILD}/obj_train_data")

names = []
for path in files:
    name = os.path.splitext(os.path.basename(path))[0]
    shutil.copy(path, f"{BUILD}/obj_train_data/{name}.jpg")
    src = os.path.join(LABELS_DIR, name + ".txt")
    dst = f"{BUILD}/obj_train_data/{name}.txt"
    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        open(dst, "w").close()
    names.append(name)

open(f"{BUILD}/obj.names", "w").write("car\n")
open(f"{BUILD}/obj.data", "w").write(
    "classes = 1\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n")
open(f"{BUILD}/train.txt", "w").write(
    "".join(f"data/obj_train_data/{n}.jpg\n" for n in names))

with zipfile.ZipFile("cvat_images.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for n in names:
        z.write(f"{BUILD}/obj_train_data/{n}.jpg", f"{n}.jpg")
with zipfile.ZipFile("cvat_import.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in ("obj.names", "obj.data", "train.txt"):
        z.write(f"{BUILD}/{f}", f)
    for n in names:
        for ext in ("jpg", "txt"):
            z.write(f"{BUILD}/obj_train_data/{n}.{ext}", f"obj_train_data/{n}.{ext}")
print(f"Готово: {len(names)} кадров -> cvat_images.zip, cvat_import.zip")