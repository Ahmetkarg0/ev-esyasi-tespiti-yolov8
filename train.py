import argparse
import csv
import json
import platform
import shutil
import subprocess
import time

import torch
from ultralytics import YOLO

from config import PROJECT_DIR, RESULTS_DIR, TRAIN_CFG, WEIGHTS_PATH


def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def thermal_guard(max_temp, resume_temp):
    state = {"n": 0, "paused": 0.0}

    def on_batch_end(trainer):
        state["n"] += 1
        if state["n"] % 50 or gpu_temp() < max_temp:
            return
        t0 = time.time()
        while gpu_temp() > resume_temp:
            time.sleep(5)
        state["paused"] += time.time() - t0

    return on_batch_end, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/coco9/coco9.yaml")
    ap.add_argument("--project", default=str(PROJECT_DIR / "runs"))
    ap.add_argument("--name", default="yolov8s_coco9")
    ap.add_argument("--epochs", type=int, default=TRAIN_CFG["epochs"])
    ap.add_argument("--batch", type=int, default=TRAIN_CFG["batch"])
    ap.add_argument("--workers", type=int, default=TRAIN_CFG["workers"])
    ap.add_argument("--resume", default="")
    ap.add_argument("--max-temp", type=int, default=0, help="Laptop için GPU sıcaklık sınırı (°C), 0 = kapalı")
    args = ap.parse_args()

    cfg = dict(TRAIN_CFG, epochs=args.epochs, batch=args.batch, workers=args.workers)
    guard = None
    t0 = time.time()
    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
    else:
        model = YOLO(cfg["model"])
        if args.max_temp:
            cb, guard = thermal_guard(args.max_temp, args.max_temp - 10)
            model.add_callback("on_train_batch_end", cb)
        model.train(data=args.data, project=args.project, name=args.name, exist_ok=True, plots=True,
                    **{k: v for k, v in cfg.items() if k != "model"})
    elapsed = time.time() - t0

    save_dir = model.trainer.save_dir
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(save_dir / "weights" / "best.pt", WEIGHTS_PATH)

    with open(save_dir / "results.csv") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    if "time" in rows[-1]:
        elapsed = max(elapsed, float(rows[-1]["time"]))
    best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))

    info = {
        "train_time_sec": round(elapsed, 1),
        "train_time_human": time.strftime("%H:%M:%S", time.gmtime(elapsed)),
        "epochs_completed": len(rows),
        "best_epoch": int(float(best["epoch"])),
        "best_val_mAP50": round(float(best["metrics/mAP50(B)"]), 4),
        "best_val_mAP50_95": round(float(best["metrics/mAP50-95(B)"]), 4),
        "thermal_pause_sec": round(guard["paused"], 1) if guard else 0,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "hyperparameters": cfg,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "train_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
