import argparse
import json
import random
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import CLASSES, RESULTS_DIR

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
NEGATIVE_CLASSES = ["apple", "banana", "carrot", "broccoli", "donut", "sports ball", "cake"]


def download(url, dst: Path, retries=3):
    if dst.exists() and dst.stat().st_size > 0:
        return True
    tmp = dst.with_suffix(dst.suffix + ".part")
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            tmp.replace(dst)
            return True
        except Exception:
            tmp.unlink(missing_ok=True)
    return False


def get_annotations(root: Path):
    ann_dir = root / "annotations"
    if (ann_dir / "instances_val2017.json").exists():
        return ann_dir
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "annotations.zip"
    print("Anotasyonlar indiriliyor (~241 MB)...")
    if not download(ANN_URL, zip_path):
        raise RuntimeError("Anotasyonlar indirilemedi.")
    with zipfile.ZipFile(zip_path) as z:
        for name in ("annotations/instances_train2017.json", "annotations/instances_val2017.json"):
            z.extract(name, root)
    zip_path.unlink()
    return ann_dir


def load_subset(json_path: Path):
    print(f"Okunuyor: {json_path.name}")
    data = json.loads(json_path.read_text())
    id2name = {c["id"]: c["name"] for c in data["categories"]}
    name2yolo = {c: i for i, c in enumerate(CLASSES)}
    images = {im["id"]: im for im in data["images"]}
    labels, cls_in_img, all_cats = defaultdict(list), defaultdict(set), defaultdict(set)

    for a in data["annotations"]:
        name = id2name[a["category_id"]]
        all_cats[a["image_id"]].add(name)
        if name not in name2yolo or a.get("iscrowd", 0):
            continue
        im = images[a["image_id"]]
        x, y, w, h = a["bbox"]
        if w <= 1 or h <= 1:
            continue
        W, H = im["width"], im["height"]
        c = name2yolo[name]
        labels[a["image_id"]].append(f"{c} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
        cls_in_img[a["image_id"]].add(c)
    return images, labels, cls_in_img, all_cats


def select_train(cls_in_img, max_n, seed):
    ids = list(cls_in_img)
    if max_n <= 0 or len(ids) <= max_n:
        return ids
    random.Random(seed).shuffle(ids)
    quota = max_n // len(CLASSES)
    freq = Counter(c for s in cls_in_img.values() for c in s)
    selected, have = [], Counter()
    chosen = set()
    for c in sorted(range(len(CLASSES)), key=lambda k: freq[k]):
        for i in ids:
            if have[c] >= quota:
                break
            if c in cls_in_img[i] and i not in chosen:
                chosen.add(i)
                selected.append(i)
                have.update(cls_in_img[i])
    selected += [i for i in ids if i not in chosen][: max_n - len(selected)]
    return selected[:max_n]


def select_negatives(all_cats, n, seed):
    ours = set(CLASSES)
    ids = [i for i, cats in all_cats.items() if cats & set(NEGATIVE_CLASSES) and not cats & ours]
    random.Random(seed).shuffle(ids)
    return ids[:n]


def write_split(split, ids, images, labels, out: Path, threads):
    img_dir, lbl_dir = out / "images" / split, out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    keep = {Path(images[i]["file_name"]).stem for i in ids}
    for p in list(img_dir.iterdir()) + list(lbl_dir.iterdir()):
        if p.stem not in keep:
            p.unlink()
    print(f"[{split}] {len(ids)} görüntü indiriliyor...")
    ok = []
    with ThreadPoolExecutor(threads) as ex:
        futs = {ex.submit(download, images[i]["coco_url"], img_dir / images[i]["file_name"]): i for i in ids}
        for k, f in enumerate(as_completed(futs), 1):
            if f.result():
                ok.append(futs[f])
            if k % 1000 == 0:
                print(f"  {k}/{len(ids)}")
    for i in ok:
        (lbl_dir / f"{Path(images[i]['file_name']).stem}.txt").write_text("\n".join(labels.get(i, [])))
    if len(ok) < len(ids):
        print(f"  Uyarı: {len(ids) - len(ok)} görüntü indirilemedi.")
    return ok


def class_stats(ids, labels):
    cnt = Counter(int(line.split()[0]) for i in ids for line in labels.get(i, []))
    return {c: cnt[k] for k, c in enumerate(CLASSES)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/coco9")
    ap.add_argument("--max-train", type=int, default=8000, help="0 = hepsi")
    ap.add_argument("--negatives", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()

    out = Path(args.out)
    ann_dir = get_annotations(out)

    tr_images, tr_labels, tr_cls, tr_cats = load_subset(ann_dir / "instances_train2017.json")
    train_ids = select_train(tr_cls, args.max_train, args.seed)
    neg_ids = select_negatives(tr_cats, args.negatives, args.seed)
    del tr_cats

    va_images, va_labels, va_cls, _ = load_subset(ann_dir / "instances_val2017.json")
    va_ids = sorted(va_cls)
    random.Random(args.seed).shuffle(va_ids)
    half = len(va_ids) // 2

    train_ids = write_split("train", train_ids + neg_ids, tr_images, tr_labels, out, args.threads)
    val_ids = write_split("val", va_ids[:half], va_images, va_labels, out, args.threads)
    test_ids = write_split("test", va_ids[half:], va_images, va_labels, out, args.threads)

    yaml = [f"path: {out.as_posix()}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml += [f"  {i}: {c}" for i, c in enumerate(CLASSES)]
    (out / "coco9.yaml").write_text("\n".join(yaml) + "\n")

    n_neg = sum(1 for i in train_ids if i not in tr_labels)
    stats = {
        "train": {"images": len(train_ids), "negatives": n_neg, "instances": class_stats(train_ids, tr_labels)},
        "val": {"images": len(val_ids), "instances": class_stats(val_ids, va_labels)},
        "test": {"images": len(test_ids), "instances": class_stats(test_ids, va_labels)},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"Hazır: {out / 'coco9.yaml'}")


if __name__ == "__main__":
    main()
