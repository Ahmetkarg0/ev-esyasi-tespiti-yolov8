import argparse
import time

import gradio as gr
import pandas as pd
from PIL import Image, ImageOps
from ultralytics import YOLO

from config import CLASSES, CLASSES_TR, CONF_THRES, IOU_THRES, TEST_PHOTOS_DIR, WEIGHTS_PATH
from postprocess import color_filter

model = YOLO(str(WEIGHTS_PATH))
model.predict(Image.new("RGB", (640, 640)), verbose=False)
COLUMNS = ["Sınıf", "Türkçe", "Güven", "Kutu (x1, y1, x2, y2)"]


def detect(image, conf, iou, use_filter):
    if image is None:
        return None, pd.DataFrame(columns=COLUMNS), "Lütfen bir fotoğraf yükleyin."
    image = ImageOps.exif_transpose(image).convert("RGB")

    t0 = time.perf_counter()
    r = model.predict(image, conf=conf, iou=iou, imgsz=640, verbose=False)[0]
    removed = 0
    if use_filter:
        r, removed = color_filter(r, image)
    total_ms = (time.perf_counter() - t0) * 1000

    rows = [[r.names[int(c)], CLASSES_TR[r.names[int(c)]], round(float(s), 3), ", ".join(str(int(v)) for v in b)]
            for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
    df = pd.DataFrame(rows, columns=COLUMNS)

    if rows:
        summary = "**Tespit edilenler:** " + ", ".join(f"{k} ×{v}" for k, v in df["Türkçe"].value_counts().items())
    else:
        summary = "**Nesne bulunamadı.** Güven eşiğini düşürmeyi deneyin."
    if removed:
        summary += f"\n\nRenk filtresi portakal rengi olmayan {removed} 'orange' tespitini eledi."
    summary += f"\n\nSüre: **{total_ms:.0f} ms** (model çıkarımı {r.speed['inference']:.0f} ms)"
    return Image.fromarray(r.plot()[..., ::-1]), df, summary


examples = []
if TEST_PHOTOS_DIR.exists():
    examples = [[str(p), CONF_THRES, IOU_THRES, True] for p in sorted(TEST_PHOTOS_DIR.glob("*.jpg"))]

with gr.Blocks(title="Ev Eşyası Tespiti") as demo:
    gr.Markdown("# Ev Eşyası Tespiti (YOLOv8s)\nTanınan sınıflar: " + ", ".join(CLASSES_TR[c] for c in CLASSES))
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Fotoğraf", sources=["upload", "webcam", "clipboard"])
            conf = gr.Slider(0.05, 0.95, value=CONF_THRES, step=0.05, label="Güven eşiği")
            iou = gr.Slider(0.1, 0.95, value=IOU_THRES, step=0.05, label="NMS IoU eşiği")
            flt = gr.Checkbox(value=True, label="Portakal renk filtresi")
            btn = gr.Button("Tespit et", variant="primary")
        with gr.Column():
            out_img = gr.Image(type="pil", label="Sonuç")
            out_txt = gr.Markdown()
    out_df = gr.Dataframe(label="Tespitler", interactive=False)

    btn.click(detect, [inp, conf, iou, flt], [out_img, out_df, out_txt])
    if examples:
        gr.Examples(examples, [inp, conf, iou, flt], [out_img, out_df, out_txt], fn=detect, label="Test fotoğrafları")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    demo.launch(share=args.share, server_port=args.port)
