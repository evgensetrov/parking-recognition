# Превращает полигоны и маски из экспорта CVAT в прямоугольники (по крайним точкам)
# и собирает архив аннотаций, который можно загрузить обратно в CVAT для доработки.
#
# 1) В CVAT: меню задачи (⋮) -> Export task dataset -> формат «CVAT for images 1.1»
#    (также поддерживаются «COCO 1.0» и «Ultralytics YOLO Segmentation»),
#    «Save images» можно выключить. Положите архив в ./cvat_export.zip
#    или распакуйте в папку ./cvat_export.
# 2) Запустите:  python polygons_to_boxes.py
# 3) В CVAT: меню задачи (⋮) -> Upload annotations -> тот же формат, что при экспорте
#    (для XML это «CVAT 1.1») -> выберите ./annotations_boxes.zip
#
# Формат результата совпадает с форматом входа, картинки в архив не попадают.

import glob
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter

# ------------------------------- Настройки ---------------------------------
IN_ZIP = "./cvat_export.zip"      # архив из CVAT (если есть — распаковывается в IN_DIR)
IN_DIR = "./cvat_export"
OUT_ZIP = "./annotations_boxes.zip"
MIN_BOX_PX = 2                    # рамки уже этого (по любой стороне) считаются мусором
KEEP_ORIGINAL = False             # True — оставить исходные полигоны рядом с рамками (только XML)
PREVIEW_COUNT = 3                 # картинки с рамками для проверки -> ./preview_boxes
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

stats = Counter()


def bbox_from_points(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    if w and h:
        x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
        y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
    return x1, y1, x2, y2


def too_small(box):
    return box[2] - box[0] < MIN_BOX_PX or box[3] - box[1] < MIN_BOX_PX


# ------------------------------- CVAT XML ----------------------------------
def shape_to_box(sh):
    """bbox фигуры CVAT (polygon / mask) или None."""
    if sh.tag == "polygon":
        pts = [tuple(map(float, p.split(","))) for p in sh.get("points").split(";")]
        return bbox_from_points(pts)
    if sh.tag == "mask":
        l, t = float(sh.get("left")), float(sh.get("top"))
        return l, t, l + float(sh.get("width")), t + float(sh.get("height"))
    return None


def convert_xml(src, dst):
    tree = ET.parse(src)
    root = tree.getroot()
    stats["рамок уже было"] += sum(1 for _ in root.iter("box"))
    sizes = {}
    for im in root.iter("image"):
        sizes[im] = (float(im.get("width", 0)), float(im.get("height", 0)))

    for parent in list(root.iter()):
        w, h = sizes.get(parent, (0, 0))
        for idx, sh in enumerate(list(parent)):
            if sh.tag not in ("polygon", "mask"):
                continue
            box = shape_to_box(sh)
            kind = sh.tag
            if box is None:
                continue
            box = clip_box(box, w, h)
            if too_small(box):
                parent.remove(sh)
                stats["удалено (слишком мелкие)"] += 1
                continue
            new = ET.Element("box")
            for k, v in sh.attrib.items():
                if k not in ("points", "rle", "left", "top", "width", "height"):
                    new.set(k, v)
            new.set("xtl", f"{box[0]:.2f}")
            new.set("ytl", f"{box[1]:.2f}")
            new.set("xbr", f"{box[2]:.2f}")
            new.set("ybr", f"{box[3]:.2f}")
            if parent.tag == "image" and new.get("source") is None:
                new.set("source", "manual")
            for child in list(sh):          # атрибуты объекта сохраняем
                new.append(child)
            new.tail = sh.tail
            parent.remove(sh)
            parent.insert(idx, new)
            if KEEP_ORIGINAL:
                parent.insert(idx + 1, sh)
            stats[f"{kind} -> рамка"] += 1
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tree.write(dst, encoding="utf-8", xml_declaration=True)


# ------------------------------- COCO --------------------------------------
def convert_coco(src, dst):
    data = json.load(open(src, encoding="utf-8"))
    imgs = {i["id"]: i for i in data["images"]}
    keep = []
    for a in data["annotations"]:
        im = imgs.get(a["image_id"], {})
        seg = a.get("segmentation")
        if isinstance(seg, list) and seg and isinstance(seg[0], list):
            flat = [v for poly in seg for v in poly]
            box = bbox_from_points(list(zip(flat[0::2], flat[1::2])))
            kind = "polygon"
        elif a.get("bbox") and len(a["bbox"]) == 4:
            x, y, w, h = a["bbox"]          # маска (RLE) или готовая рамка: bbox уже есть
            box = (x, y, x + w, y + h)
            kind = "mask" if seg else "box"
        else:
            keep.append(a)
            continue
        box = clip_box(box, im.get("width"), im.get("height"))
        if too_small(box):
            stats["удалено (слишком мелкие)"] += 1
            continue
        x1, y1, x2, y2 = box
        a["bbox"] = [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]
        a["segmentation"] = [[x1, y1, x2, y1, x2, y2, x1, y2]]
        a["area"] = round((x2 - x1) * (y2 - y1), 2)
        a["iscrowd"] = 0
        stats[f"{kind} -> рамка"] += 1
        keep.append(a)
    data["annotations"] = keep
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(data, open(dst, "w", encoding="utf-8"), ensure_ascii=False)


# ------------------------------- YOLO txt ----------------------------------
def convert_yolo_txt(src, dst):
    lines = []
    for line in open(src):
        p = line.split()
        if len(p) < 5:
            continue
        v = list(map(float, p[1:]))
        if len(v) == 4:
            lines.append(line.strip())
            stats["рамок уже было"] += 1
            continue
        x1, y1, x2, y2 = bbox_from_points([(v[i], v[i + 1]) for i in range(0, len(v) - 1, 2)])
        x1, x2 = max(0.0, x1), min(1.0, x2)
        y1, y2 = max(0.0, y1), min(1.0, y2)
        if x2 <= x1 or y2 <= y1:
            stats["удалено (слишком мелкие)"] += 1
            continue
        lines.append(f"{p[0]} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
        stats["polygon -> рамка"] += 1
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w").write("\n".join(lines) + ("\n" if lines else ""))


# ------------------------------- Основное ----------------------------------
def is_cvat_xml(path):
    return "<annotations" in open(path, encoding="utf-8", errors="ignore").read(3000)


def is_coco(path):
    try:
        d = json.load(open(path, encoding="utf-8"))
        return isinstance(d, dict) and "images" in d and "annotations" in d
    except Exception:
        return False


def make_preview(xml_or_none):
    """Рисует рамки на нескольких картинках из экспорта (если картинки есть)."""
    try:
        import cv2
    except ImportError:
        return
    images = {os.path.basename(p): p for p in glob.glob(os.path.join(IN_DIR, "**", "*.*"), recursive=True)
              if p.lower().endswith(IMG_EXT)}
    if not images or not xml_or_none:
        return
    os.makedirs("./preview_boxes", exist_ok=True)
    n = 0
    for im in ET.parse(xml_or_none).getroot().iter("image"):
        name = os.path.basename(im.get("name"))
        if name not in images or n >= PREVIEW_COUNT:
            continue
        img = cv2.imread(images[name])
        for b in im.iter("box"):
            cv2.rectangle(img, (int(float(b.get("xtl"))), int(float(b.get("ytl")))),
                          (int(float(b.get("xbr"))), int(float(b.get("ybr")))), (0, 255, 0), 3)
        k = 1600 / img.shape[1]
        cv2.imwrite(f"./preview_boxes/{os.path.splitext(name)[0]}.jpg",
                    cv2.resize(img, None, fx=k, fy=k))
        n += 1


def main():
    if os.path.exists(IN_ZIP):
        shutil.rmtree(IN_DIR, ignore_errors=True)
        with zipfile.ZipFile(IN_ZIP) as z:
            z.extractall(IN_DIR)
        print(f"Распаковал {IN_ZIP} -> {IN_DIR}")
    if not os.path.isdir(IN_DIR):
        sys.exit(f"Не найден ни {IN_ZIP}, ни папка {IN_DIR}. Положите экспорт из CVAT.")

    work = "./_boxes_build"
    shutil.rmtree(work, ignore_errors=True)
    files = [p for p in glob.glob(os.path.join(IN_DIR, "**", "*"), recursive=True)
             if os.path.isfile(p) and not p.lower().endswith(IMG_EXT)]
    xml_out = None
    fmt = None
    for p in files:
        rel = os.path.relpath(p, IN_DIR)
        dst = os.path.join(work, rel)
        low = p.lower()
        if low.endswith(".xml") and is_cvat_xml(p):
            fmt = "CVAT for images 1.1 (XML)"
            convert_xml(p, dst)
            xml_out = dst
        elif low.endswith(".json") and is_coco(p):
            fmt = "COCO"
            convert_coco(p, dst)
        elif low.endswith(".txt") and os.sep + "labels" + os.sep in os.sep + rel:
            fmt = fmt or "YOLO (txt)"
            convert_yolo_txt(p, dst)
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(p, dst)          # служебные файлы (data.yaml, train.txt и т.п.)
    if fmt is None:
        sys.exit("Не нашёл аннотаций поддерживаемого формата "
                 "(CVAT XML, COCO, YOLO txt в папке labels).")

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in glob.glob(os.path.join(work, "**", "*"), recursive=True):
            if os.path.isfile(p):
                z.write(p, os.path.relpath(p, work))
    make_preview(xml_out)
    shutil.rmtree(work)
    print(f"Формат: {fmt}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"Готово: {OUT_ZIP}")


if __name__ == "__main__":
    main()