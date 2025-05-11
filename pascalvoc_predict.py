import torch
from pathlib import Path
import cv2
import os

source_dir = "VOCdevkit/VOC2012/JPEGImages"
output_dir = "VOCdevkit/VOC2012/predictions_pascal"
conf_threshold = 0.5
img_size = 640

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

os.makedirs(output_dir, exist_ok=True)

try:

    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True, force_reload=True)
    model.to(device)
    model.conf = conf_threshold

    for img_path in Path(source_dir).glob("*.jpg"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model(img, size=img_size)

        txt_path = os.path.join(output_dir, f"{img_path.stem}.txt")
        with open(txt_path, "w") as f:
            for *xyxy, conf, cls in results.xyxy[0]:
                f.write(
                    f"{int(cls)} {float(conf):.6f} {float(xyxy[0]):.1f} {float(xyxy[1]):.1f} {float(xyxy[2]):.1f} {float(xyxy[3]):.1f}\n")

    print(f"Predictions successfully saved to {output_dir}")

except Exception as e:
    print(f"Error: {str(e)}")
    # if "CUDA" in str(e):
    #     print("Try using CPU instead by setting device='cpu'")