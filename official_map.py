# See cocoapi/PythonAPI/pycocoEvalDemo.ipynb
import argparse
import cv2
import numpy as np
import os
import tempfile
import torch
import torchvision.ops as ops
import yaml
import json
import warnings
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor, Compose
from tqdm import tqdm
from PIL import Image

warnings.filterwarnings(action="ignore")


LABEL_MAPPING = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def parse_args():
    parser = argparse.ArgumentParser("Evaluate MS COCO mAP")
    parser.add_argument("--data", help="Path to the dataset", default=".")
    parser.add_argument("--model", help="Model name", default="ultralytics/yolov5:yolov5x")
    parser.add_argument("--split", help="Dataset part", default="val2017")
    parser.add_argument("--batch-size", help="Batch size", type=int, default=32)
    parser.add_argument("--image-size", help="Image size", type=int, default=640)
    parser.add_argument("--num-workers", help="Image size", type=int, default=8)
    parser.add_argument("--iou-threshold", help="Evaluation IoU threshold", type=float, default=0.5)
    parser.add_argument("--dump-predictions", help="Dump evaluation results to a json file with a given name")
    parser.add_argument("--top-k", type=int, default=1, help="Predict top K labels for each bbox")
    return parser.parse_args()


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        w, h = image.size
        scale = self.size / max(w, h)
        return image.resize((int(round(w * scale)), int(round(h * scale))),
                            resample=Image.BILINEAR)


class AddPadding:
    def __init__(self, pad_value=114 / 255.0):
        self.pad_value = pad_value

    def __call__(self, image, targets):
        c, h, w = image.shape
        size = max(h, w)
        result = torch.full((c, size, size), self.pad_value, dtype=torch.float, device=image.device)
        top = (size - h) // 2
        left = (size - w) // 2
        result[:, top:top + h, left:left + w] = image
        new_targets = []
        for t in targets:
            bx, by, bw, bh = t["bbox"]
            new_targets.append({
                "image_id": t["image_id"],
                "bbox": (bx + left, by + top, bw, bh),
                "category_id": t["category_id"]
            })
        return top, left, result, new_targets


class Transform:
    def __init__(self, image_size):
        self.image_size = image_size
        self.transform = Compose([
            Resize(image_size),
            ToTensor()
        ])
        self.padding = AddPadding()

    def __call__(self, image, targets):
        orig_w, orig_h = image.size
        image = self.transform(image)
        top, left, image, targets = self.padding(image, targets)
        image_id = targets[0]["image_id"] if targets else None
        for t in targets:
            assert t["image_id"] == image_id
        targets = {
            "image_id": image_id,
            "offset": (top, left),
            "scale": image.shape[1] / max(orig_w, orig_h),
            "detections": [{"bbox": t["bbox"],
                            "category_id": t["category_id"]}
                           for t in targets]
        }
        return image, targets


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, 0), targets


def result2coco(anno, results, dump_predictions=None):
    clean = []
    for r in results:
        clean.append({
            "image_id": r["image_id"],
            "bbox": r["bbox"],
            "category_id": r["category_id"]
        })
        if "score" in r:
            clean[-1]["score"] = r["score"]
    if dump_predictions:
        with open(dump_predictions, "w") as fp:
            json.dump(clean, fp)
        return anno.loadRes(dump_predictions)
    else:
        with tempfile.NamedTemporaryFile(mode="w") as fp:
            json.dump(clean, fp)
            fp.flush()
            return anno.loadRes(fp.name)


def evaluate(model, data_root, data_split,
             batch_size, image_size,
             iou_threshold,
             num_workers,
             dump_predictions=None,
             min_confidence=0.001,
             topk=1):
    use_cuda = torch.cuda.is_available()

    transforms = Transform(image_size)
    dataset = CocoDetection(os.path.join(data_root, data_split),
                            os.path.join(data_root, "annotations", f"instances_{data_split}.json"),
                            transforms=transforms)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         pin_memory=use_cuda, num_workers=num_workers,
                                         collate_fn=collate_fn)

    model.eval()
    if use_cuda:
        model.cuda()
    print(f"The dataset contains {len(dataset)} images.")

    results = []
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(loader)):
            if use_cuda:
                images = images.cuda()
            predictions = model(images)  # (B, N, 4 [bbox] + 1 [confidence] + C).

            bboxes = predictions[:, :, :4].clone()  # (B, N, 4), Center XY + WH.
            bboxes[:, :, :2] -= bboxes[:, :, 2:] / 2  # (B, N, 4, Corner XY + WH.
            confidences = predictions[:, :, 4]  # (B, N).
            label_probs = predictions[:, :, 5:]  # (B, N, C).

            predictions_mask = confidences >= min_confidence  # (B, N).

            for i in range(len(images)):
                image_id = dataset.ids[batch_idx * batch_size + i]
                if targets[i]["image_id"] is not None:
                    assert targets[i]["image_id"] == image_id
                scale = targets[i]["scale"]
                top, left = targets[i]["offset"]
                offset = torch.tensor((left, top, 0, 0))

                image_mask = predictions_mask[i] >= min_confidence  # (K).
                image_bboxes = bboxes[i][image_mask]  # (K, 4).
                image_confidences = confidences[i][image_mask]  # (K).
                image_label_probs = label_probs[i][image_mask]  # (K, C).

                image_bboxes_xyxy = torch.cat([image_bboxes[:, :2], image_bboxes[:, :2] + image_bboxes[:, 2:]], 1)
                nms_indices = ops.nms(image_bboxes_xyxy, image_confidences, iou_threshold)
                bboxes_cpu = image_bboxes[nms_indices].cpu()  # (K, 4).
                label_confidences_cpu = (image_label_probs[nms_indices] * image_confidences[nms_indices][:, None]).cpu()  # (K, C).
                for j in range(len(nms_indices)):
                    #score, category = label_confidences_cpu[j].max(0)
                    top_scores, top_categories = label_confidences_cpu[j].topk(1)

                                        # добавляем оба предсказания в результаты
                    for score, category in zip(top_scores, top_categories):
                        results.append({
                            "image_id": image_id,
                            "bbox": ((bboxes_cpu[j] - offset) / scale).tolist(),
                            "category_id": LABEL_MAPPING[category.item()],
                            "score": score.item(),
                            #"label_probs": label_confidences_cpu[j]
                        })

    anno = COCO(os.path.join(data_root, "annotations", f"instances_{data_split}.json"))
    coco = COCOeval(anno,
                    result2coco(anno, results, dump_predictions=dump_predictions),
                    "bbox")
    coco.evaluate()
    coco.accumulate()
    coco.summarize()
    return {"mAP-0": coco.stats[0],
            "mAP-50": coco.stats[1],
            "mAP-75": coco.stats[2]}


if __name__ == "__main__":
    args = parse_args()
    repo, name = args.model.split(":")
    model = torch.hub.load(repo, name, pretrained=True)
    metrics = evaluate(model, args.data, args.split,
                       batch_size=args.batch_size,
                       image_size=args.image_size,
                       iou_threshold=args.iou_threshold,
                       num_workers=args.num_workers,
                       dump_predictions=args.dump_predictions,
                       topk=args.top_k)
    print("Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    # Список изображений
    #imgIds = cocoGt.getImgIds()
