# Шаг 2. Превращает экспорт из CVAT (полигоны, маски, прямоугольники) в датасет YOLO
# с обычными прямоугольными рамками для обучения.
#
# Распакуйте экспорт CVAT в папку ./cvat_export. Поддерживаются форматы экспорта
# (формат определяется автоматически):
#   1) «CVAT for images 1.1»          — annotations.xml   (рекомендуется)
#   2) «COCO 1.0»                     — файл .json
#   3) «Ultralytics YOLO Segmentation» / «YOLO Detection» — .txt рядом с картинками
# При экспорте включите «Save images».
#
# Результат: ./dataset (images/train|val, labels/train|val, dataset.yaml)
# и ./dataset/preview с несколькими картинками для проверки рамок.

import glob
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter

import cv2
from PIL import Image

# ------------------------------- Настройки ---------------------------------
EXPORT_DIR = "./cvat_export"
DATASET = "./dataset"
VAL_FRACTION = 0.15      # доля кадров для проверки (если кадров >= 10)
SEED = 42
MIN_BOX_PX = 8           # рамки меньше этого размера (по любой стороне) отбрасываются
KEEP_LABELS = None       # None — брать все метки как «car»; либо список, например ["car", "truck"]
PREVIEW_COUNT = 3        # сколько картинок с нарисованными рамками сохранить для проверки
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ------------------------------- Чтение форматов ---------------------------
def bbox_from_points(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def parse_cvat_xml(xml_path):
    """-> {имя_файла: (ширина, высота, [(метка, x1, y1, x2, y2), ...])} в пикселях."""
    res = {}
    for im in ET.parse(xml_path).getroot().iter("image"):
        name = os.path.basename(im.get("name"))
        w, h = int(float(im.get("width"))), int(float(im.get("height")))
        items = []
        for sh in im:
            label = sh.get("label")
            if sh.tag == "box":
                items.append((label, float(sh.get("xtl")), float(sh.get("ytl")),
                              float(sh.get("xbr")), float(sh.get("ybr"))))
            elif sh.tag == "polygon":
                pts = [tuple(map(float, p.split(","))) for p in sh.get("points").split(";")]
                items.append((label,) + bbox_from_points(pts))
            elif sh.tag == "mask":  # маски (например, от SAM): рамка задана напрямую
                l, t = float(sh.get("left")), float(sh.get("top"))
                items.append((label, l, t, l + float(sh.get("width")), t + float(sh.get("height"))))
        res[name] = (w, h, items)
    return res


def parse_coco(json_path):
    data = json.load(open(json_path, encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    imgs = {i["id"]: i for i in data["images"]}
    res = {os.path.basename(i["file_name"]): (int(i["width"]), int(i["height"]), [])
           for i in imgs.values()}
    for a in data["annotations"]:
        im = imgs[a["image_id"]]
        label = cats.get(a["category_id"], str(a["category_id"]))
        seg = a.get("segmentation")
        if a.get("bbox") and len(a["bbox"]) == 4:
            x, y, w, h = a["bbox"]
            box = (x, y, x + w, y + h)
        elif isinstance(seg, list) and seg:
            flat = [v for poly in seg for v in poly]
            box = bbox_from_points(list(zip(flat[0::2], flat[1::2])))
        else:
            continue
        res[os.path.basename(im["file_name"])][2].append((label,) + tuple(box))
    return res


def parse_yolo_txt(images):
    """Метки YOLO: строка «класс x y w h» или «класс x1 y1 x2 y2 ...» (нормированные)."""
    res = {}
    for name, path in images.items():
        txt = os.path.splitext(path)[0] + ".txt"
        if not os.path.exists(txt):
            txt = txt.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
        if not os.path.exists(txt):
            continue
        with Image.open(path) as im:
            w, h = im.size
        items = []
        for line in open(txt):
            p = line.split()
            if len(p) < 5:
                continue
            c, v = p[0], list(map(float, p[1:]))
            if len(v) == 4:
                cx, cy, bw, bh = v
                box = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
            else:
                box = bbox_from_points([(v[i] * w, v[i + 1] * h) for i in range(0, len(v) - 1, 2)])
            items.append((c,) + box)
        res[name] = (w, h, items)
    return res


# ------------------------------- Основное ----------------------------------
def main():
    images = {}
    for p in glob.glob(os.path.join(EXPORT_DIR, "**", "*.*"), recursive=True):
        if p.lower().endswith(IMG_EXT):
            name = os.path.basename(p)
            if name in images:
                print(f"Предупреждение: повторяется имя файла {name} — использую первый")
                continue
            images[name] = p
    if not images:
        raise SystemExit(f"В {EXPORT_DIR} нет картинок. При экспорте из CVAT включите «Save images».")

    xmls = [p for p in glob.glob(os.path.join(EXPORT_DIR, "**", "*.xml"), recursive=True)
            if "<annotations" in open(p, encoding="utf-8", errors="ignore").read(2000)]
    jsons = []
    for p in glob.glob(os.path.join(EXPORT_DIR, "**", "*.json"), recursive=True):
        try:
            d = json.load(open(p, encoding="utf-8"))
            if isinstance(d, dict) and "images" in d and "annotations" in d:
                jsons.append(p)
        except Exception:
            pass

    if xmls:
        print("Формат: CVAT for images 1.1 (XML)")
        ann = parse_cvat_xml(xmls[0])
    elif jsons:
        print("Формат: COCO")
        ann = parse_coco(jsons[0])
    else:
        print("Формат: YOLO (txt)")
        ann = parse_yolo_txt(images)
    if not ann:
        raise SystemExit("Аннотации не найдены. Проверьте формат экспорта.")

    # --- перевод в YOLO-рамки
    label_stats, dropped = Counter(), Counter()
    samples = []  # (путь картинки, имя, [строки меток], [рамки в пикселях])
    for name, (w, h, items) in ann.items():
        if name not in images:
            print(f"Нет картинки для {name} — пропускаю")
            continue
        lines, boxes = [], []
        for label, x1, y1, x2, y2 in items:
            if KEEP_LABELS is not None and label not in KEEP_LABELS:
                dropped["метка не из KEEP_LABELS"] += 1
                continue
            x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))   # обрезаем по кадру
            y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
            if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
                dropped["слишком маленькая рамка"] += 1
                continue
            lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            boxes.append((x1, y1, x2, y2))
            label_stats[label] += 1
        samples.append((images[name], name, lines, boxes))
    if not samples:
        raise SystemExit("Не удалось сопоставить аннотации с картинками.")

    # --- разбиение на train / val
    random.Random(SEED).shuffle(samples)
    n_val = max(1, round(len(samples) * VAL_FRACTION)) if len(samples) >= 10 else 0
    val, train = samples[:n_val], samples[n_val:]
    if not val:
        val = train
        print("ВНИМАНИЕ: кадров меньше 10 — проверка идёт на обучающих, "
              "реальную точность это не показывает.")

    shutil.rmtree(DATASET, ignore_errors=True)
    for split, items in (("train", train), ("val", val)):
        os.makedirs(f"{DATASET}/images/{split}")
        os.makedirs(f"{DATASET}/labels/{split}")
        for path, name, lines, _ in items:
            shutil.copy(path, f"{DATASET}/images/{split}/{name}")
            with open(f"{DATASET}/labels/{split}/{os.path.splitext(name)[0]}.txt", "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
    with open(f"{DATASET}/dataset.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(DATASET)}\n"
                "train: images/train\nval: images/val\nnames:\n  0: car\n")

    # --- картинки для проверки
    os.makedirs(f"{DATASET}/preview", exist_ok=True)
    for path, name, _, boxes in samples[:PREVIEW_COUNT]:
        img = cv2.imread(path)
        if img is None:
            continue
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        k = 1600 / img.shape[1]
        cv2.imwrite(f"{DATASET}/preview/{os.path.splitext(name)[0]}.jpg", cv2.resize(img, None, fx=k, fy=k))

    total = sum(label_stats.values())
    print(f"Датасет готов: кадров train={len(train)}, val={len(val)}, всего рамок: {total}")
    print("Метки в разметке:", dict(label_stats))
    if len(label_stats) > 1 and KEEP_LABELS is None:
        print("ВНИМАНИЕ: найдено несколько меток, все они стали классом «car». "
              "Если среди них есть лишние (например, person), задайте KEEP_LABELS "
              "в начале файла и запустите заново.")
    if dropped:
        print("Отброшено:", dict(dropped))
    print(f"Проверьте рамки на картинках в {DATASET}/preview")


if __name__ == "__main__":
    main()