import torch
import torchvision
import os
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DetrImageProcessor, DetrForObjectDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import warnings
import json
import datasets

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, feature_extractor):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return pixel_values, target

# def visualize_image(dataset, image_id):
#     image_info = dataset.coco.loadImgs(image_id)[0]
#     image = Image.open(os.path.join(dataset.root, image_info['file_name']))
#
#     annotations = dataset.coco.imgToAnns[image_id]
#     draw = ImageDraw.Draw(image, "RGBA")
#
#     cats = dataset.coco.cats
#     id2label = {k: v['name'] for k, v in cats.items()}
#
#     for annotation in annotations:
#         box = annotation['bbox']
#         class_idx = annotation['category_id']
#         x, y, w, h = tuple(box)
#         draw.rectangle((x, y, x + w, y + h), outline='red', width=1)
#         draw.text((x, y), id2label[class_idx], fill='white')
#
#     image.show()

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = feature_extractor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }


# Класс для оценки
class SimpleCocoEvaluator:
    def __init__(self, coco_gt, iou_types):
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.coco_eval = {iou_type: COCOeval(coco_gt, iouType=iou_type)
                          for iou_type in iou_types}
        self.predictions = []

    def update(self, predictions):
        # Преобразуем предсказания в правильный формат
        formatted_predictions = []
        for img_id, pred in predictions.items():
            for i in range(len(pred['scores'])):
                formatted_predictions.append({
                    'image_id': img_id,
                    'category_id': int(pred['labels'][i]),
                    'bbox': [float(x) for x in pred['boxes'][i].tolist()],
                    'score': float(pred['scores'][i])
                })
        self.predictions.extend(formatted_predictions)

    def synchronize_between_processes(self):
        pass

    def accumulate(self):
        for iou_type in self.iou_types:
            if len(self.predictions) == 0:
                continue
            coco_dt = self.coco_gt.loadRes(self.predictions)
            coco_eval = self.coco_eval[iou_type]
            coco_eval.cocoDt = coco_dt
            coco_eval.evaluate()
            coco_eval.accumulate()

    def summarize(self):
        for iou_type in self.iou_types:
            print(f"IoU metric: {iou_type}")
            self.coco_eval[iou_type].summarize()


# Основная функция
def evaluate():
    global feature_extractor

    # Проверка зависимостей
    try:
        import timm
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "timm"])

    # Инициализация модели
    feature_extractor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
    model.eval()

    # Пути к данным
    img_folder = 'val2017'
    ann_file = 'annotations/instances_val2017.json'

    # Датасет
    dataset = CocoDetection(
        img_folder=img_folder,
        ann_file=ann_file,
        feature_extractor=feature_extractor
    )

    # # Визуализация
    # image_ids = dataset.coco.getImgIds()
    # image_id = image_ids[np.random.randint(0, len(image_ids))]
    # print(f'Visualizing image ID: {image_id}')
    # visualize_image(dataset, image_id)

    # Даталоадер
    test_dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=1)

    # Инициализация оценщика
    base_ds = dataset.coco
    iou_types = ['bbox']
    coco_evaluator = SimpleCocoEvaluator(base_ds, iou_types)

    # Оценка
    print("Running evaluation...")
    for idx, batch in enumerate(tqdm(test_dataloader)):
        # if idx > 50:  # Ограничение для быстрой проверки
        #     break

        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = feature_extractor.post_process_object_detection(
            outputs,
            target_sizes=orig_target_sizes,
            threshold=0.0
        )

        res = {target['image_id'].item(): output for target, output in zip(labels, results)}
        coco_evaluator.update(res)

    # Результаты
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return {"mAP-0": coco_evaluator.stats[0],
            "mAP-50": coco_evaluator.stats[1],
            "mAP-75": coco_evaluator.stats[2]}

if __name__ == "__main__":
    evaluate()

#https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/DETR/Evaluating_DETR_on_COCO_validation_2017.ipynb#scrollTo=lAl61HW-ciqe&uniqifier=1

#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.015
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.039
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.010
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.006
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.029
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.039
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.058
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.070
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.011
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.177
# err:
# Traceback (most recent call last):
#   File "C:\Users\Admin\PycharmProjects\cocoyolov5\detr2.py", line 184, in <module>
#     evaluate()
#   File "C:\Users\Admin\PycharmProjects\cocoyolov5\detr2.py", line 179, in evaluate
#     return {"mAP-0": coco_evaluator.stats[0],
# AttributeError: 'SimpleCocoEvaluator' object has no attribute 'stats'
