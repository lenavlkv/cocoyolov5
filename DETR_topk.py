import argparse
import os
import torch
import warnings
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.datasets import CocoDetection
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
from torchvision.ops import box_convert
import torchvision.transforms as T
import time

start_time = time.time()

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
                            T.Resize(image_size),
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
            "orig_size": (orig_w, orig_h),
            "scale": image.shape[1] / max(orig_w, orig_h),
            "detections": [{"bbox": t["bbox"],
                            "category_id": t["category_id"]}
                           for t in targets]
        }
        return image, targets

def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, 0), targets

def evaluate(model, data_root, data_split,
             batch_size, image_size,
             num_workers,
             min_confidence=0.001,
             topk=1):
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

            target_sizes = torch.tensor([(t["orig_size"][1], t["orig_size"][0]) for t in targets], device=images.device)

            prob = nn.functional.softmax(pred_logits, -1)
            scores, labels = prob[..., :-1].topk(topk, dim=-1)

            predictions_mask = scores.max(-1).values >= min_confidence

            boxes = box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xywh")
            img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            boxes = boxes * scale_fct[:, None, :]

            for i in range(len(images)):
                image_id = dataset.ids[batch_idx * batch_size + i]

                mask = predictions_mask[i]
                filtered_scores = scores[i][mask]
                filtered_labels = labels[i][mask]
                filtered_boxes = boxes[i][mask].unsqueeze(1).repeat(1, topk, 1)

                image_results = []
                for box_idx in range(filtered_boxes.shape[0]):
                    for k in range(topk):
                        image_results.append({
                            'image_id': image_id,
                            'category_id': int(filtered_labels[box_idx, k].item()),
                            'score': float(filtered_scores[box_idx, k].item()),
                            'bbox': [float(x.item()) for x in filtered_boxes[box_idx, 0]]
                        })
                results.extend(image_results)

    coco_pred = anno.loadRes(results)
    coco_eval = COCOeval(anno, coco_pred, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {"mAP-0": coco_eval.stats[0],
            "mAP-50": coco_eval.stats[1],
            "mAP-75": coco_eval.stats[2]}

if __name__ == "__main__":
    args = parse_args()
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)

    metrics = evaluate(model, args.data, args.split,
                       batch_size=args.batch_size,
                       image_size=args.image_size,
                       num_workers=args.num_workers,
                       topk=args.top_k)
    print("Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    print(f"Time: {time.time() - start_time:.2f}s")