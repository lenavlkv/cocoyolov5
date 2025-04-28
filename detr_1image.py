import argparse
import os
import tempfile
import torch
import torchvision.ops as ops
import json
import warnings
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor, Compose
from tqdm import tqdm
from PIL import Image
import io
import pathlib
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import torch.nn as nn
import numpy as np
from torchvision.ops import box_convert
import torchvision.transforms as T



warnings.filterwarnings(action="ignore")


def parse_args():
    parser = argparse.ArgumentParser("Evaluate MS COCO mAP")
    parser.add_argument("--data", help="Path to the dataset", default=".")
    parser.add_argument("--split", help="Dataset part", default="val2017")
    parser.add_argument("--batch-size", help="Batch size", type=int, default=1)
    parser.add_argument("--image-size", help="Image size", type=int, default=800)
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

class Transform:
    def __init__(self, image_size):
        self.image_size = image_size
        self.transform = T.Compose([
                            T.Resize(800),
                            T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

    def __call__(self, image, targets):
        orig_w, orig_h = image.size
        image = self.transform(image)
        image_id = targets[0]["image_id"] if targets else None
        for t in targets:
            assert t["image_id"] == image_id
        targets = {
            "image_id": image_id,
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
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as fp:
            json.dump(clean, fp)
            fp.flush()
            return anno.loadRes(fp.name)



def evaluate(model, data_root, data_split,
             batch_size, image_size,
             iou_threshold,
             num_workers,
             dump_predictions=None,
             min_confidence=0.5,
             topk=1,
             target_image_id=69138):
    use_cuda = torch.cuda.is_available()

    transforms = Transform(image_size)
    dataset = CocoDetection(os.path.join(data_root, data_split),
                            os.path.join(data_root, "annotations", f"instances_{data_split}.json"),
                            transforms=transforms)

    anno = COCO(os.path.join(data_root, "annotations", f"instances_{data_split}.json"))
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
            predictions = model(images)
            pred_logits = predictions['pred_logits']
            pred_boxes = predictions['pred_boxes']

            prob = nn.functional.softmax(pred_logits, -1)
            scores, labels = prob[..., :-1].max(-1)

            target_sizes = torch.tensor([(image_size, image_size)] * images.shape[0], device=images.device)

            predictions_mask = scores >= min_confidence

            boxes = box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xywh")
            img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            boxes = boxes * scale_fct[:, None, :]

            for i in range(len(images)):
                image_id = dataset.ids[batch_idx * batch_size + i]
                if image_id != target_image_id:
                    continue

                print(f"\nProcessing image ID: {image_id}")

                # GT
                ann_ids = anno.getAnnIds(imgIds=image_id)
                gt_anns = anno.loadAnns(ann_ids)
                gt_bboxes = [ann["bbox"] for ann in gt_anns] #xywh
                gt_cats = [ann["category_id"] for ann in gt_anns]

                print("\nGround Truth:")
                for bbox, cat_id in zip(gt_bboxes, gt_cats):
                    print(f"  Category: {cat_id}, BBox: {bbox}")

                # Predicted
                mask = predictions_mask[i]
                filtered_scores = scores[i][mask]
                filtered_labels = labels[i][mask]
                filtered_boxes = boxes[i][mask]
                # print(filtered_labels)
                # print(filtered_scores)
                # print(filtered_boxes)

                # filtered_scores = filtered_scores[i].topk(topk)

                # Results
                image_results = []
                for score, label, box in zip(filtered_scores, filtered_labels, filtered_boxes):
                    image_results.append({
                        'image_id': image_id,
                        'category_id': int(label.item()),
                        'score': float(score.item()),
                        'bbox': [float(x.item()) for x in box] #xywh
                    })

                print("Predictions:")
                for res in image_results:
                    print(f"  Category: {res['category_id']}, Score: {res['score']:.4f}, BBox: {res['bbox']}")

                # top_2 = sorted([a for a in zip(filtered_scores, key=lambda w: w[1], reverse=True)][:2])
                # print('top_2',len(top_2))


                results.extend(image_results)

    return results


if __name__ == "__main__":
    args = parse_args()
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)

    # Вызываем evaluate с указанием целевого image_id
    evaluate(model, args.data, args.split,
             batch_size=args.batch_size,
             image_size=args.image_size,
             iou_threshold=args.iou_threshold,
             num_workers=args.num_workers,
             dump_predictions=args.dump_predictions,
             topk=args.top_k,
             target_image_id=69138)