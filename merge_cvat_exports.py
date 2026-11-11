#!/usr/bin/env python3
"""Добавляет доразмеченную партию кадров в общий набор для обучения.

Общий набор — папка ./cvat_export (annotations.xml + images/). Её же читает 2_make_dataset.py.

  python merge_cvat_exports.py --new-xml batch2/annotations.xml --new-images cvat_pack/images

  --new-xml     annotations.xml из экспорта CVAT («CVAT for images 1.1»)
  --new-images  папка images/ из export_for_cvat.py (те же кадры, что загружались в CVAT)
Если кадр с таким именем уже есть в наборе, он заменяется новой разметкой.
"""
import argparse
import copy
import os
import shutil
import sys
import xml.etree.ElementTree as ET

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--into", default="./cvat_export")
p.add_argument("--new-xml", required=True)
p.add_argument("--new-images", required=True)
a = p.parse_args()

new_root = ET.parse(a.new_xml).getroot()
base_xml = os.path.join(a.into, "annotations.xml")
os.makedirs(os.path.join(a.into, "images"), exist_ok=True)
if os.path.exists(base_xml):
    tree = ET.parse(base_xml)
    root = tree.getroot()
else:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = new_root.find("meta")
    if meta is not None:
        root.append(copy.deepcopy(meta))
    tree = ET.ElementTree(root)

existing = {os.path.basename(im.get("name")): im for im in root.iter("image")}
next_id = max([int(im.get("id", 0)) for im in existing.values()] + [-1]) + 1
added = replaced = skipped = 0
for im in new_root.iter("image"):
    name = os.path.basename(im.get("name"))
    src = os.path.join(a.new_images, name)
    if not os.path.exists(src):
        print(f"Нет картинки {name} в {a.new_images} — кадр пропущен")
        skipped += 1
        continue
    if name in existing:
        root.remove(existing[name])
        replaced += 1
    else:
        added += 1
    new_im = copy.deepcopy(im)
    new_im.set("id", str(next_id))
    new_im.set("name", name)
    next_id += 1
    root.append(new_im)
    shutil.copy(src, os.path.join(a.into, "images", name))

ET.indent(tree, space="  ")
tree.write(base_xml, encoding="utf-8", xml_declaration=True)
total = sum(1 for _ in root.iter("image"))
boxes = sum(1 for _ in root.iter("box"))
print(f"Добавлено {added}, заменено {replaced}, пропущено {skipped}. В наборе кадров: {total}, рамок: {boxes}")
if skipped:
    sys.exit(1)
