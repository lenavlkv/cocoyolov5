import argparse
import json
import os
import tempfile
import warnings
from collections import defaultdict
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO

warnings.filterwarnings("ignore")

LABEL_MAPPING = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88,
    89, 90
]


def parse_args():
    parser = argparse.ArgumentParser("Evaluate MS COCO mAP with YOLOv8")
    parser.add_argument("--data", help="Path to COCO dataset root", default=".")
    parser.add_argument("--weights", help="Path to YOLOv8 model weights", required=True)
    parser.add_argument("--split", help="Dataset split to evaluate", default="val2017")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="IoU threshold for NMS")
    parser.add_argument("--output", help="Path to save predictions JSON file")
    parser.add_argument("--top-k", type=int, default=2, help="Number of top predictions to consider for each box")
    return parser.parse_args()


class CocoEvaluator:
    def __init__(self, args):
        self.args = args
        self.model = YOLO(args.weights)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Prepare dataset
        ann_file = os.path.join(args.data, "annotations", f"instances_{args.split}.json")
        img_dir = os.path.join(args.data, args.split)
        self.dataset = CocoDetection(img_dir, ann_file, transform=ToTensor())
        self.coco_gt = COCO(ann_file)

    def evaluate(self):
        results = []
        confusion_pairs = defaultdict(int)  # Для анализа ошибок

        for img_idx in tqdm(range(len(self.dataset)), desc="Evaluating"):

            # Get image and targets
            img, targets = self.dataset[img_idx]
            img_id = self.dataset.ids[img_idx]

            # Get original image dimensions
            if targets and 'width' in targets[0] and 'height' in targets[0]:
                orig_w = targets[0]['width']
                orig_h = targets[0]['height']
            else:
                orig_w, orig_h = img.shape[2], img.shape[1]

            # Convert tensor to PIL Image
            img_pil = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

            # Run inference
            with torch.no_grad():
                outputs = self.model(img_pil, imgsz=self.args.img_size, verbose=False)[0]

            # Get class probabilities (shape: [num_boxes, num_classes])
            cls_probs = outputs.boxes.cls_prob.cpu() if hasattr(outputs.boxes, 'cls_prob') else None

            # Process predictions
            for i, (box, conf, cls_id) in enumerate(zip(outputs.boxes.xywhn.cpu(),
                                                        outputs.boxes.conf.cpu(),
                                                        outputs.boxes.cls.cpu().int())):
                if conf < self.args.conf_thres:
                    continue

                # Convert normalized coordinates to absolute
                x_center, y_center, w, h = box
                x = (x_center - w / 2) * orig_w
                y = (y_center - h / 2) * orig_h
                w = w * orig_w
                h = h * orig_h

                # Get top-k predictions
                if cls_probs is not None and self.args.top_k > 1:
                    # Get top-k classes and their probabilities
                    topk_probs, topk_classes = torch.topk(cls_probs[i], k=self.args.top_k)

                    # Add each top prediction to results
                    for prob, cls_idx in zip(topk_probs, topk_classes):
                        results.append({
                            "image_id": img_id,
                            "bbox": [x.item(), y.item(), w.item(), h.item()],
                            "category_id": LABEL_MAPPING[cls_idx.item()],
                            "score": (conf * prob).item()  # Combine box confidence and class probability
                        })

                        # For confusion analysis (only if topk > 1)
                        if self.args.top_k > 1 and targets:
                            # Get ground truth classes for this image
                            gt_classes = [t['category_id'] for t in targets if 'category_id' in t]

                            # Check if second prediction matches any GT while first doesn't
                            if (len(topk_classes) > 1 and
                                    LABEL_MAPPING[topk_classes[0].item()] not in gt_classes and
                                    LABEL_MAPPING[topk_classes[1].item()] in gt_classes):
                                pair = (
                                LABEL_MAPPING[topk_classes[1].item()], LABEL_MAPPING[topk_classes[0].item()])
                                confusion_pairs[pair] += 1
                else:
                    # Default single prediction
                    results.append({
                        "image_id": img_id,
                        "bbox": [x.item(), y.item(), w.item(), h.item()],
                        "category_id": LABEL_MAPPING[cls_id.item()],
                        "score": conf.item()
                    })



        if self.args.output:
            with open(self.args.output, 'w') as f:
                json.dump(results, f)
            coco_preds = self.coco_gt.loadRes(self.args.output)
        else:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                json.dump(results, f)
                f.flush()
                coco_preds = self.coco_gt.loadRes(f.name)

        # Run COCO evaluation
        coco_eval = COCOeval(self.coco_gt, coco_preds, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return {
            "mAP@0.5:0.95": coco_eval.stats[0],
            "mAP@0.5": coco_eval.stats[1],
            "mAP@0.75": coco_eval.stats[2],
            "confusion_pairs": dict(sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True))
        }

if __name__ == "__main__":
    args = parse_args()
    print(f"Evaluating YOLOv8 on COCO {args.split} set...")
    print(f"Model weights: {args.weights}")
    print(f"Image size: {args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Confidence threshold: {args.conf_thres}")
    print(f"Top-K predictions: {args.top_k}")

    evaluator = CocoEvaluator(args)
    metrics = evaluator.evaluate()

    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        if k != "confusion_pairs":
            print(f"{k}: {v:.4f}")

    if args.top_k > 1 and metrics["confusion_pairs"]:
        print("\nMost Common Confusion Pairs (correct, wrong):")
        for (correct, wrong), count in list(metrics["confusion_pairs"].items())[:10]:
            print(f"{correct} mistaken as {wrong}: {count} times")


# python YOLOv8_topk.py --weights yolov8x.pt  --top-k 1
#Evaluation Metrics:
# mAP@0.5:0.95: 0.4803
# mAP@0.5: 0.6161
# mAP@0.75: 0.5246