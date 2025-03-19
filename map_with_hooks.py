import torch
from pycocotools.coco import COCO
import cv2
import numpy as np
import torchvision.ops as ops
import os
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import ops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
from collections import defaultdict
import time
from tqdm import tqdm

start_time = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка модели YOLO
model = YOLO('yolov5x.pt').to(device)

# Загрузка аннотаций COCO
annFile = 'annotations/instances_val2017.json'
coco = COCO(annFile)

# Список изображений для обработки
imgIds = coco.getImgIds()

# Названия классов COCO
coco_class_names = [coco.loadCats(i)[0]['name'] for i in coco.getCatIds()]

# Названия классов модели YOLO
yolo_class_names = model.names

# Сопоставление названий классов COCO и модели
class_mapping = {coco_name: yolo_name for coco_name, yolo_name in zip(coco_class_names, yolo_class_names)}

class SaveIO:
    """Simple PyTorch hook to save the output of a nn.module."""
    def __init__(self):
        self.input = None
        self.output = None

    def __call__(self, module, module_in, module_out):
        self.input = module_in
        self.output = module_out

def load_and_prepare_model(model_path):
    detect = None
    cv2_hooks = None
    cv3_hooks = None
    detect_hook = SaveIO()
    for i, module in enumerate(model.model.modules()):
        if type(module) is Detect:
            module.register_forward_hook(detect_hook)
            detect = module

            cv2_hooks = [SaveIO() for _ in range(module.nl)]
            cv3_hooks = [SaveIO() for _ in range(module.nl)]
            for i in range(module.nl):
                module.cv2[i].register_forward_hook(cv2_hooks[i])
                module.cv3[i].register_forward_hook(cv3_hooks[i])
            break
    input_hook = SaveIO()
    model.model.register_forward_hook(input_hook)

    hooks = [input_hook, detect, detect_hook, cv2_hooks, cv3_hooks]

    return model, hooks

def is_text_file(file_path):
    # Check if the file extension indicates a text file
    text_extensions = ['.txt']  # Add more extensions if needed
    return any(file_path.lower().endswith(ext) for ext in text_extensions)

def is_image_file(file_path):
    # Check if the file extension indicates an image file
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more extensions if needed
    return any(file_path.lower().endswith(ext) for ext in image_extensions)

def write_json(results):
    # Create a list to store the predictions data
    predictions = []

    for result in results:
        image_id = os.path.basename(result['image_id'])
        max_category_id = result['activations'].index(max(result['activations']))
        category_id = max_category_id
        bbox = result['bbox']
        score = max(result['activations'])
        activations = result['activations']

        prediction = {
            'image_id': image_id,
            'category_id': category_id,
            'bbox': bbox,
            'score': score,
            'activations': activations
        }

        predictions.append(prediction)

    # Write the predictions list to a JSON file
    with open('predictions.json', 'w') as f:
        json.dump(predictions, f)

# Функция для вычисления IoU
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    inter_x1 = max(x1, x1_gt)
    inter_y1 = max(y1, y1_gt)
    inter_x2 = min(x2, x2_gt)
    inter_y2 = min(y2, y2_gt)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou

# Apply Non-Maximum Suppression
def nms(boxes, iou_threshold=0.7):
    sorted_boxes = sorted(boxes, key=lambda x: max(x['activations']), reverse=True)
    selected_boxes = []

    delete_idxs = []
    for i, box0 in enumerate(sorted_boxes):
        for j, box1 in enumerate(sorted_boxes):
            if i < j and compute_iou(box0['bbox'], box1['bbox']) > iou_threshold:
                delete_idxs.append(j)

    delete_idxs.reverse()

    filtered_boxes = [box for idx, box in enumerate(sorted_boxes) if idx not in delete_idxs]

    return filtered_boxes

def results_predict(img_path, model, hooks, threshold=0.5, iou=0.7, save_image=False, category_mapping=None):
    input_hook, detect, detect_hook, cv2_hooks, cv3_hooks = hooks

    model(img_path)

    shape = detect_hook.input[0][0].shape  # BCHW
    x = []
    for i in range(detect.nl):
        x.append(torch.cat((cv2_hooks[i].output, cv3_hooks[i].output), 1))
    x_cat = torch.cat([xi.view(shape[0], detect.no, -1) for xi in x], 2)
    box, cls = x_cat.split((detect.reg_max * 4, detect.nc), 1)

    batch_idx = 0
    xywh_sigmoid = detect_hook.output[0][batch_idx]
    all_logits = cls[batch_idx]

    img_shape = input_hook.input[0].shape[2:]
    orig_img_shape = model.predictor.batch[1][batch_idx].shape[:2]

    # compute predictions
    boxes = []
    for i in range(xywh_sigmoid.shape[-1]):
        x0, y0, x1, y1, *class_probs_after_sigmoid = xywh_sigmoid[:, i]
        x0, y0, x1, y1 = ops.scale_boxes(img_shape, np.array([x0.cpu(), y0.cpu(), x1.cpu(), y1.cpu()]), orig_img_shape)
        logits = all_logits[:, i]

        boxes.append({
            'image_id': img_path,
            'bbox': [x0.item(), y0.item(), x1.item(), y1.item()],  # xyxy
            'bbox_xywh': [(x0.item() + x1.item()) / 2, (y0.item() + y1.item()) / 2, x1.item() - x0.item(), y1.item() - y0.item()],
            'logits': logits.cpu().tolist(),
            'activations': [p.item() for p in class_probs_after_sigmoid]
        })

    boxes_for_nms = torch.stack([
        torch.tensor([*b['bbox_xywh'], *b['activations'], *b['activations'], *b['logits']]) for b in boxes
    ], dim=1).unsqueeze(0)

    nms_results = ops.non_max_suppression(boxes_for_nms, conf_thres=threshold, iou_thres=iou, nc=detect.nc)[0]

    boxes = []
    for b in range(nms_results.shape[0]):
        box = nms_results[b, :]
        x0, y0, x1, y1, conf, cls, *acts_and_logits = box
        activations = acts_and_logits[:detect.nc]
        logits = acts_and_logits[detect.nc:]
        box_dict = {
            'bbox': [x0.item(), y0.item(), x1.item(), y1.item()],  # xyxy
            'bbox_xywh': [(x0.item() + x1.item()) / 2, (y0.item() + y1.item()) / 2, x1.item() - x0.item(), y1.item() - y0.item()],
            'best_conf': conf.item(),
            'best_cls': cls.item(),
            'image_id': img_path,
            'activations': [p.item() for p in activations],
            'logits': [p.item() for p in logits]
        }
        boxes.append(box_dict)

    return boxes

def run_predict(input_path, model, hooks, score_threshold=0.5, iou_threshold=0.7, save_image=False, save_json=False, category_mapping=None):
    use_txt_input = False

    if is_text_file(input_path):
        use_txt_input = True

    if use_txt_input:
        with open(input_path, 'r') as f:
            img_paths = f.read().splitlines()
    else:
        img_paths = [input_path]

    all_results = []

    for img_path in img_paths:
        results = results_predict(img_path, model, hooks, score_threshold, iou=iou_threshold, save_image=save_image, category_mapping=category_mapping)

        all_results.extend(results)

    if save_json:
        write_json(all_results)

    return all_results

def main(img_id):
    model_path = 'yolov5x.pt'
    img = coco.loadImgs(img_id)[0]
    img_path = f'val2017/{img["file_name"]}'
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    threshold = 0.3
    nms_threshold = 0.7

    # Загрузка модели
    model, hooks = load_and_prepare_model(model_path)

    # Получаем названия классов из модели YOLO
    classes = model.names  # Добавлено определение переменной classes

    # Выполнение предсказания
    results = run_predict(img_path, model, hooks, save_image=True, score_threshold=threshold, iou_threshold=nms_threshold)

    # Загрузка аннотаций для текущего изображения
    annIds = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(annIds)

    iou_results = defaultdict(list)

    # Сохраняем предсказания и ground truth для дальнейшей оценки
    for ann in anns:
        gt_box = ann['bbox']  # Координаты [x, y, width, height]
        gt_x1, gt_y1, gt_width, gt_height = gt_box
        gt_x2 = gt_x1 + gt_width
        gt_y2 = gt_y1 + gt_height
        gt_box = [gt_x1, gt_y1, gt_x2, gt_y2]  # Преобразуем в формат [x1, y1, x2, y2]
        cat_id = ann['category_id']
        gt_category = coco.loadCats(cat_id)[0]['name']
        iou_results['gt'].append((gt_box, gt_category))

    mismatched = []

    for i, pred in enumerate(results):
        top_2 = sorted([a for a in zip(classes, pred['activations'])], key=lambda w: w[1], reverse=True)[:2]
        x0, y0, x1, y1 = map(int, pred['bbox'])
        confidence = pred['best_conf']
        class_id = int(pred['best_cls'])
        label = classes[class_id]  # Название класса из модели
        pred_box = [x0, y0, x1, y1]
        second_class_name, second_confidence = top_2[1]

        max_iou = 0
        best_gt = None
        for gt_box, gt_category in iou_results['gt']:
            iou = compute_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                best_gt = gt_category  # Название класса из COCO

        # Сравнение по названиям классов
        if max_iou >= 0.5 and label == best_gt:
            iou_results[label].append(1)  # True positive
        else:
            iou_results[label].append(0)  # False positive

    #     # Если второй класс оказался правильным
    #     if second_class_name == best_gt and label != best_gt:
    #         iou_results[label].append(1)
    #         mismatched.append(best_gt)
    #         mismatched.append(label)
    #         mismatched.append(confidence)
    #         mismatched.append(second_class_name)
    #         mismatched.append(second_confidence)
    # if mismatched:
    #     print('MISSED: ', mismatched)

    return iou_results

all_results = defaultdict(list)
# Общее количество изображений
total_images = len(imgIds)

for img_id in tqdm(imgIds, desc="Обработка изображений", unit="img"):
    img_results = main(img_id)
    for label, results in img_results.items():
        if label != 'gt':  # Только классы, а не "gt"
            all_results[label].extend(results)

def compute_map(all_results):
    aps = {}
    for label, result in all_results.items():
        tp = np.array(result)
        fp = 1 - tp
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)

        # Если нет TP, AP = 0
        if np.sum(tp) == 0:
            aps[label] = 0.0
            continue

        recall = cumsum_tp / np.sum(tp)
        precision = cumsum_tp / (cumsum_tp + cumsum_fp)

        # Интерполяция для расчета AP
        recall_interpolated = np.linspace(0, 1, 101)
        precision_interpolated = np.interp(recall_interpolated, recall[::-1], precision[::-1])

        ap = np.mean(precision_interpolated)
        aps[label] = ap

    mAP = np.mean(list(aps.values()))  # Среднее значение по всем классам
    return aps, mAP

# Вычисление mAP для всех классов
aps, mAP = compute_map(all_results)
print(f"Mean Average Precision (mAP): {mAP:.4f}")
for label, ap in aps.items():
    print(f"AP for class {label}: {ap:.4f}")

print(f"Time: {time.time() - start_time:.2f}s")