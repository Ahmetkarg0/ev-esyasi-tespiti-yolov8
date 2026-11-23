import argparse
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps
from ultralytics import YOLO

from config import CONF_THRES, IOU_THRES, RESULTS_DIR, TEST_PHOTOS_DIR, WEIGHTS_PATH
from postprocess import color_filter

# sınıf -> (en az, en fazla); None = üst sınır yok
EXPECTED = {
    "sise": ("Mutfak tezgâhında şeffaf plastik su şişesi", {"bottle": (1, 1)}),
    "kasik_catal": ("Granit tezgâhta metal kaşık ve çatal", {"spoon": (1, 1), "fork": (1, 1)}),
    "meyveler": ("İki portakal ve bir kivi (kivi COCO'da yok)", {"orange": (2, 2)}),
    "kitaplar": ("Masada üst üste kitaplar ve defterler", {"book": (2, None)}),
    "bardak_kupa": ("Cam bardak ve seramik kupa", {"cup": (2, 2)}),
    "sandalye": ("Kahverengi deri, krom ayaklı sandalye", {"chair": (1, 1)}),
    "koltuk": ("Gri kumaş kanepe, üzerinde yeşil yastıklar", {"couch": (1, 1)}),
}


def fmt_expected(exp):
    return ", ".join(f"{c} ×{lo}+" if hi is None else f"{c} ×{lo}" for c, (lo, hi) in exp.items())


def fmt_counts(counts):
    return ", ".join(f"{c} ×{n}" for c, n in counts.most_common()) or "(tespit yok)"


def judge(exp, counts):
    ok = sum(lo <= counts.get(c, 0) and (hi is None or counts.get(c, 0) <= hi) for c, (lo, hi) in exp.items())
    missing = any(counts.get(c, 0) == 0 for c in exp)
    extra = any(c not in exp for c in counts)
    if ok == len(exp) and not extra:
        return "Başarılı"
    return "Başarısız" if missing else "Kısmen başarılı"


def make_grid(paths, out_path, cell=640, cols=4):
    rows = math.ceil(len(paths) / cols)
    grid = Image.new("RGB", (cols * cell, rows * cell), "white")
    for k, p in enumerate(paths):
        im = Image.open(p)
        im.thumbnail((cell, cell))
        grid.paste(im, ((k % cols) * cell + (cell - im.width) // 2, (k // cols) * cell + (cell - im.height) // 2))
    grid.save(out_path, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(WEIGHTS_PATH))
    ap.add_argument("--photos", default=str(TEST_PHOTOS_DIR))
    ap.add_argument("--conf", type=float, default=CONF_THRES)
    args = ap.parse_args()

    out_dir = RESULTS_DIR / "test_photos"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    files = sorted(p for p in Path(args.photos).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    records, pred_paths = [], []
    for f in files:
        img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        raw = model.predict(img, conf=args.conf, iou=IOU_THRES, imgsz=640, verbose=False)[0]
        r, removed = color_filter(raw, img)

        Image.fromarray(raw.plot()[..., ::-1]).save(out_dir / f"{f.stem}_ham.jpg", quality=90)
        pred_path = out_dir / f"{f.stem}_pred.jpg"
        Image.fromarray(r.plot()[..., ::-1]).save(pred_path, quality=90)
        pred_paths.append(pred_path)

        desc, exp = EXPECTED.get(f.stem, ("", {}))
        raw_counts = Counter(raw.names[int(c)] for c in raw.boxes.cls)
        counts = Counter(r.names[int(c)] for c in r.boxes.cls)
        records.append(dict(
            dosya=f.name, aciklama=desc, beklenen=fmt_expected(exp),
            ham_tahmin=fmt_counts(raw_counts), tahmin=fmt_counts(counts), renk_filtresi_eledi=removed,
            durum_ham=judge(exp, raw_counts), durum=judge(exp, counts),
            sure_ms=round(sum(raw.speed.values()), 1),
            tespitler=[dict(sinif=r.names[int(c)], guven=round(float(s), 3)) for c, s in zip(r.boxes.cls, r.boxes.conf)],
        ))
        print(f"{f.name:16s} beklenen: {records[-1]['beklenen']:24s} tahmin: {records[-1]['tahmin']:28s} {records[-1]['durum']}")

    make_grid(pred_paths, out_dir / "grid.jpg")
    (out_dir / "sonuclar.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["| # | Fotoğraf | İçerik | Beklenen | Model tahmini (güven) | Sonuç |", "|---|---|---|---|---|---|"]
    for k, rec in enumerate(records, 1):
        det = ", ".join(f"{d['sinif']} ({d['guven']:.2f})" for d in rec["tespitler"]) or "(tespit yok)"
        if rec["renk_filtresi_eledi"]:
            det += f" — renk filtresi {rec['renk_filtresi_eledi']} 'orange' tespitini eledi (ham: {rec['ham_tahmin']})"
        lines.append(f"| {k} | {rec['dosya']} | {rec['aciklama']} | {rec['beklenen']} | {det} | {rec['durum']} |")
    n_ok = sum(r["durum"] == "Başarılı" for r in records)
    lines.append(f"\nTam başarılı: **{n_ok}/{len(records)}** (conf ≥ {args.conf})")
    (out_dir / "sonuclar.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
