import numpy as np
from PIL import Image

ORANGE_HUE = (10, 65)
MIN_SAT, MIN_VAL = 0.5, 0.5
MIN_FRACTION = 0.4


def orange_fraction(hsv: np.ndarray, box) -> float:
    x1, y1, x2, y2 = (int(v) for v in box)
    w, h = x2 - x1, y2 - y1
    crop = hsv[y1 + h // 4: y2 - h // 4, x1 + w // 4: x2 - w // 4].reshape(-1, 3) / 255.0
    if len(crop) == 0:
        return 1.0
    hue = crop[:, 0] * 360
    ok = (hue >= ORANGE_HUE[0]) & (hue <= ORANGE_HUE[1]) & (crop[:, 1] >= MIN_SAT) & (crop[:, 2] >= MIN_VAL)
    return float(ok.mean())


def color_filter(result, image: Image.Image):
    boxes = result.boxes
    if len(boxes) == 0:
        return result, 0
    hsv = np.asarray(image.convert("HSV"))
    keep = [
        result.names[int(c)] != "orange" or orange_fraction(hsv, b) >= MIN_FRACTION
        for c, b in zip(boxes.cls.tolist(), boxes.xyxy.tolist())
    ]
    return result[np.array(keep)], keep.count(False)
