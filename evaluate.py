"""Test kümesi değerlendirmesi: P, R, F1, mAP@0.5, mAP@0.5:0.95, ortalama IoU, tahmin süresi.

    python evaluate.py --data /content/coco9/coco9.yaml --cpu-timing 50
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

from config import CLASSES, CONF_THRES, IOU_THRES, MATCH_IOU, RESULTS_DIR, WEIGHTS_PATH
from postprocess import color_filter

N = len(CLASSES)


def read_gt(path: Path, w, h):
    arr = np.loadtxt(path, ndmin=2) if path.exists() and path.stat().st_size else np.zeros((0, 5))
    c, cx, cy, bw, bh = arr.T
    box = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
    return c.astype(int), box


def match(pc, pb, pf, gc, gb):
    """Aynı sınıftaki kutuları güven sırasına göre açgözlü eşleştirir; (iou, sınıf) listesi döner."""
    out, used = [], set()
    if not len(pb) or not len(gb):
        return out
    iou = box_iou(torch.tensor(pb), torch.tensor(gb)).numpy()
    for p in np.argsort(-pf):
        cand = [(iou[p, j], j) for j in range(len(gb)) if j not in used and gc[j] == pc[p] and iou[p, j] >= MATCH_IOU]
        if cand:
            v, j = max(cand)
            used.add(j)
            out.append((float(v), int(pc[p])))
    return out


class Counts:
    def __init__(self):
        self.tp, self.fp, self.fn = np.zeros(N), np.zeros(N), np.zeros(N)
        self.ious = [[] for _ in range(N)]

    def add(self, boxes, gc, gb):
        pc = boxes.cls.cpu().numpy().astype(int)
        m = match(pc, boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), gc, gb)
        mc = [c for _, c in m]
        for v, c in m:
            self.ious[c].append(v)
        for c in range(N):
            self.tp[c] += mc.count(c)
            self.fp[c] += (pc == c).sum() - mc.count(c)
            self.fn[c] += (gc == c).sum() - mc.count(c)

    def summary(self, c=None):
        tp, fp, fn = (x.sum() if c is None else x[c] for x in (self.tp, self.fp, self.fn))
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        ious = [v for l in self.ious for v in l] if c is None else self.ious[c]
        return dict(precision=round(p, 4), recall=round(r, 4), f1=round(2 * p * r / (p + r + 1e-9), 4),
                    mean_IoU=round(float(np.mean(ious)), 4) if ious else 0.0,
                    TP=int(tp), FP=int(fp), FN=int(fn))


def cpu_timing(weights, files):
    model = YOLO(weights)
    model.predict(str(files[0]), device="cpu", verbose=False)
    t = []
    for f in files:
        t0 = time.perf_counter()
        model.predict(str(f), device="cpu", imgsz=640, verbose=False)
        t.append((time.perf_counter() - t0) * 1000)
    return round(float(np.mean(t)), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/coco9/coco9.yaml")
    ap.add_argument("--weights", default=str(WEIGHTS_PATH))
    ap.add_argument("--cpu-timing", type=int, default=0)
    args = ap.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(args.weights)

    # 1) mAP, P, R (Ultralytics doğrulayıcısı)
    m = model.val(data=args.data, split="test", imgsz=640, batch=16, iou=IOU_THRES, device=device,
                  plots=True, project=str(RESULTS_DIR), name="test_eval", exist_ok=True)
    box = m.box
    per_class = {}
    for k, ci in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(k)
        per_class[CLASSES[int(ci)]] = dict(precision=round(float(p), 4), recall=round(float(r), 4),
                                          f1=round(float(2 * p * r / (p + r + 1e-9)), 4),
                                          mAP50=round(float(ap50), 4), mAP50_95=round(float(ap), 4))
    mp, mr = float(box.mp), float(box.mr)

    # 2) conf=0.25'te IoU ve TP/FP/FN; renk filtresiyle ve filtresiz
    cfg = yaml.safe_load(open(args.data))
    img_dir = Path(cfg["path"]) / cfg["test"]
    lbl_dir = Path(cfg["path"]) / cfg["test"].replace("images", "labels")
    raw, filt, speeds = Counts(), Counts(), []
    for r in model.predict(source=str(img_dir), stream=True, conf=CONF_THRES, iou=IOU_THRES,
                           imgsz=640, device=device, verbose=False):
        h, w = r.orig_shape
        gc, gb = read_gt(lbl_dir / (Path(r.path).stem + ".txt"), w, h)
        raw.add(r.boxes, gc, gb)
        fr, _ = color_filter(r, Image.fromarray(r.orig_img[..., ::-1]))
        filt.add(fr.boxes, gc, gb)
        speeds.append(r.speed)

    for i, c in enumerate(CLASSES):
        if c in per_class:
            per_class[c]["at_conf_0.25"] = raw.summary(i)

    spd = {k: round(float(np.mean([s[k] for s in speeds])), 2) for k in ("preprocess", "inference", "postprocess")}
    oi = CLASSES.index("orange")
    results = {
        "overall": {
            "precision": round(mp, 4), "recall": round(mr, 4),
            "f1": round(2 * mp * mr / (mp + mr + 1e-9), 4),
            "mAP50": round(float(box.map50), 4), "mAP50_95": round(float(box.map), 4),
            "at_conf_0.25": raw.summary(),
        },
        "per_class": per_class,
        "orange_color_filter": {"without": raw.summary(oi), "with": filt.summary(oi)},
        "inference_ms_per_image_gpu": spd | {"total": round(sum(spd.values()), 2)},
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    if args.cpu_timing:
        files = sorted(img_dir.glob("*.jpg"))[: args.cpu_timing]
        results["inference_ms_per_image_cpu"] = cpu_timing(args.weights, files)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    lines = ["| Sınıf | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 | Ort. IoU |", "|---|---|---|---|---|---|---|"]
    for c, v in per_class.items():
        lines.append(f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['mAP50']:.3f} | "
                     f"{v['mAP50_95']:.3f} | {v['at_conf_0.25']['mean_IoU']:.3f} |")
    o = results["overall"]
    lines.append(f"| **Tümü** | **{o['precision']:.3f}** | **{o['recall']:.3f}** | **{o['f1']:.3f}** | "
                 f"**{o['mAP50']:.3f}** | **{o['mAP50_95']:.3f}** | **{o['at_conf_0.25']['mean_IoU']:.3f}** |")
    (RESULTS_DIR / "metrics_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
