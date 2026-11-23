from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

CLASSES = ["bottle", "wine glass", "cup", "fork", "spoon", "orange", "chair", "couch", "book"]

CLASSES_TR = {
    "bottle": "şişe",
    "wine glass": "şarap kadehi",
    "cup": "bardak / kupa",
    "fork": "çatal",
    "spoon": "kaşık",
    "orange": "portakal",
    "chair": "sandalye",
    "couch": "kanepe / koltuk",
    "book": "kitap",
}

TRAIN_CFG = dict(
    model="yolov8s.pt",
    imgsz=640,
    epochs=20,
    batch=16,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    cos_lr=True,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_bias_lr=0.0,
    patience=10,
    seed=42,
    workers=2,
    amp=True,
)

CONF_THRES = 0.25
IOU_THRES = 0.7     # NMS
MATCH_IOU = 0.5

WEIGHTS_PATH = PROJECT_DIR / "weights" / "best.pt"
RESULTS_DIR = PROJECT_DIR / "results"
TEST_PHOTOS_DIR = PROJECT_DIR / "test_photos"
