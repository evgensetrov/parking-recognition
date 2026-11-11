# Шаг 3. Дообучение YOLO. Запускать в Colab с GPU (T4).
from ultralytics import YOLO

model = YOLO("yolo11m.pt")  # стартуем с обычных весов
model.train(
    data="./dataset/dataset.yaml",
    imgsz=1920,       # 4K-кадр уменьшается до 1920: машины остаются ~60 px
    epochs=80,
    batch=4,          # при нехватке памяти GPU поставьте 2
    patience=20,
    degrees=3, fliplr=0.5, hsv_v=0.5,
    project="runs", name="parking", exist_ok=True,
)
print("Готово. Веса: runs/parking/weights/best.pt")
