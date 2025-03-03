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

model = YOLO('yolov5x.pt')
class_names = model.names

annFile = 'annotations/instances_val2017.json'
coco = COCO(annFile)

# Список изображений
imgIds = coco.getImgIds()

imgIds = coco.getImgIds()
img_id = imgIds[30]
img = coco.loadImgs(img_id)[0]
img_path = f'val2017/{img["file_name"]}'
img_path1 = img_path
img_cv = cv2.imread(img_path)

cl = '''0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush'''
classes = [w.split(':')[1].strip() for w in cl.split('\n')]


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
    text_extensions = ['.txt'] #, '.csv', '.json', '.xml']  # Add more extensions if needed
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


def calculate_iou(box1, box2):

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    intersect_x1 = max(x1, x2)
    intersect_y1 = max(y1, y2)
    intersect_x2 = min(x1 + w1, x2 + w2)
    intersect_y2 = min(y1 + h1, y2 + h2)

    intersect_area = max(0, intersect_x2 - intersect_x1 + 1) * max(0, intersect_y2 - intersect_y1 + 1)
    box1_area = w1 * h1
    box2_area = w2 * h2

    iou = intersect_area / float(box1_area + box2_area - intersect_area)
    return iou


# Apply Non-Maximum Suppression
def nms(boxes, iou_threshold=0.7):

    sorted_boxes = sorted(boxes, key=lambda x: max(x['activations']), reverse=True)
    selected_boxes = []

    delete_idxs = []
    for i, box0 in enumerate(sorted_boxes):
        for j, box1 in enumerate(sorted_boxes):
            if i < j and calculate_iou(box0['bbox'], box1['bbox']) > iou_threshold:
                delete_idxs.append(j)

    delete_idxs.reverse()

    filtered_boxes = [box for idx, box in enumerate(sorted_boxes) if idx not in delete_idxs]

    return filtered_boxes


def results_predict(img_path, model, hooks, threshold=0.5, iou=0.7, save_image = False, category_mapping = None):

    # unpack hooks from load_and_prepare_model()
    input_hook, detect, detect_hook, cv2_hooks, cv3_hooks = hooks

    # run inference; we don't actually need to store the results because
    # the hooks store everything we need
    model(img_path)

    # now reverse engineer the outputs to find the logits
    # see Detect.forward(): https://github.com/ultralytics/ultralytics/blob/b638c4ed9a24270a6875cdd47d9eeda99204ef5a/ultralytics/nn/modules/head.py#L22
    shape = detect_hook.input[0][0].shape  # BCHW
    x = []
    for i in range(detect.nl):
        x.append(torch.cat((cv2_hooks[i].output, cv3_hooks[i].output), 1))
    x_cat = torch.cat([xi.view(shape[0], detect.no, -1) for xi in x], 2)
    box, cls = x_cat.split((detect.reg_max * 4, detect.nc), 1)

    batch_idx = 0
    xywh_sigmoid = detect_hook.output[0][batch_idx]
    all_logits = cls[batch_idx]

    # figure out the original img shape and model img shape so we can transform the boxes
    img_shape = input_hook.input[0].shape[2:]
    orig_img_shape = model.predictor.batch[1][batch_idx].shape[:2]

    # compute predictions
    boxes = []
    for i in range(xywh_sigmoid.shape[-1]):
        x0, y0, x1, y1, *class_probs_after_sigmoid = xywh_sigmoid[:,i]
        x0, y0, x1, y1 = ops.scale_boxes(img_shape, np.array([x0.cpu(), y0.cpu(), x1.cpu(), y1.cpu()]), orig_img_shape)
        logits = all_logits[:,i]

        boxes.append({
            'image_id': img_path,
            'bbox': [x0.item(), y0.item(), x1.item(), y1.item()], # xyxy
            'bbox_xywh': [(x0.item() + x1.item())/2, (y0.item() + y1.item())/2, x1.item() - x0.item(), y1.item() - y0.item()],
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
            'bbox': [x0.item(), y0.item(), x1.item(), y1.item()], # xyxy
            'bbox_xywh': [(x0.item() + x1.item())/2, (y0.item() + y1.item())/2, x1.item() - x0.item(), y1.item() - y0.item()],
            'best_conf': conf.item(),
            'best_cls': cls.item(),
            'image_id': img_path,
            'activations': [p.item() for p in activations],
            'logits': [p.item() for p in logits]
        }
        boxes.append(box_dict)

    return boxes


def run_predict(input_path, model, hooks, score_threshold=0.5, iou_threshold=0.7, save_image = False, save_json = False, category_mapping = None):

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


def main():
    model_path = 'yolov5x.pt'
    img_path = img_path1
    threshold = 0.3
    nms_threshold = 0.7

    # Загрузка модели
    model, hooks = load_and_prepare_model(model_path)

    # Выполнение предсказания
    results = run_predict(img_path, model, hooks, save_image=True, score_threshold=threshold, iou_threshold=nms_threshold)

    print("Processed", len(results), "boxes")

    # Вывод топ-2 классов для каждого предсказания
    for i, result in enumerate(results):
        top_2 = sorted([a for a in zip(classes, result['activations'])], key=lambda w: w[1], reverse=True)[:2]
        print(f"Prediction {i + 1}: {top_2}")

    # Загрузка аннотаций для текущего изображения
    annIds = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(annIds)

    # Отрисовка ground truth на изображении
    for ann in anns:
        bbox = ann['bbox']  # COCO bbox формат: [x_min, y_min, width, height]
        x0, y0, w, h = map(int, bbox)
        x1, y1 = x0 + w, y0 + h

        # Получение имени класса для ground truth
        cat_id = ann['category_id']
        class_name = coco.loadCats(cat_id)[0]['name']

        # Отрисовка прямоугольника
        cv2.rectangle(img_cv, (x0, y0), (x1, y1), (0, 0, 255), 2)  # Красный прямоугольник для ground truth

        # Добавление подписи
        label = f"GT: {class_name}"
        cv2.putText(img_cv, label, (x0, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Отрисовка предсказаний на изображении
    for result in results:
        x0, y0, x1, y1 = map(int, result['bbox'])
        class_id = int(result['best_cls'])  # Преобразуем class_id в int
        class_name = classes[class_id]
        confidence = result['best_conf']

        # Отрисовка прямоугольника
        cv2.rectangle(img_cv, (x0, y0), (x1, y1), (0, 255, 0), 2)  # Зеленый прямоугольник для предсказаний

        # Добавление подписи для основного класса
        label_main = f"{class_name} {confidence:.2f}"
        cv2.putText(img_cv, label_main, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Получение второго класса
        top_2 = sorted([a for a in zip(classes, result['activations'])], key=lambda w: w[1], reverse=True)[:2]
        if len(top_2) > 1:
            second_class_name, second_confidence = top_2[1]
            # Добавление подписи для второго класса
            label_second = f"{second_class_name} {second_confidence:.5f}"
            cv2.putText(img_cv, label_second, (x0, y0 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)  # Желтый цвет

    # Отображение изображения с предсказаниями и ground truth
    cv2.imshow("Image with Predictions and Ground Truth", img_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()