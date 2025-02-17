import torch
from pycocotools.coco import COCO
import numpy as np
import time
import torchvision.ops as ops
import cv2

# Инициализация
start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка модели
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True).to(device)
model.eval()

# Загрузка COCO
annFile = 'annotations/instances_val2017.json'
coco = COCO(annFile)
imgIds = coco.getImgIds()  # Для обработки всего набора данных

# Сопоставление классов YOLO -> COCO
yolo_to_coco = {
    0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10,
    10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21,
    20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34,
    30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44,
    40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55,
    50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65,
    60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79,
    70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90
}

# Функция для отображения изображения с прямоугольниками и подписями
def visualize_image(img_id, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_classes):
    img = coco.loadImgs(img_id)[0]
    img_path = f'val2017/{img["file_name"]}'
    img = cv2.imread(img_path)  # Загружаем изображение с помощью OpenCV

    # Рисуем Ground Truth
    for box, cls in zip(gt_boxes, gt_classes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Красный прямоугольник
        cls_name = coco.loadCats(cls)[0]['name']
        cv2.putText(img, f'GT: {cls_name}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Подпись

    # Рисуем предсказания
    for i, (box, label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        x1, y1, x2, y2 = map(int, box)
        if i == 0:  # Первый (наиболее вероятный) класс
            color = (0, 255, 0)  # Зеленый
        else:  # Второй класс
            color = (0, 255, 255)  # Желтый

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # Прямоугольник
        cls_name = coco.loadCats(yolo_to_coco.get(int(label), -1))[0]['name']
        cv2.putText(img, f'{cls_name}: {score:.2f}', (x1, y1 - 10 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Подпись

    # Отображаем изображение
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image(img_id):
    img = coco.loadImgs(img_id)[0]
    img_path = f'val2017/{img["file_name"]}'
    img = cv2.imread(img_path)  # Загружаем изображение с помощью OpenCV

    # Предсказание
    results = model(img)  # Передаем изображение в виде numpy array
    preds = results.pred[0]
    if preds is None or len(preds) == 0:
        print("No predictions found for this image.")
        return

    # Обработка предсказаний
    boxes = preds[:, :4]
    scores = preds[:, 4]
    labels = preds[:, 5].long()

    # Выбираем два наиболее вероятных класса для каждого предсказания
    num_preds = len(scores)  # Количество всех предсказанных объектов
    top_scores, top_indices = torch.topk(scores, num_preds, largest=True)

    top_boxes = boxes[top_indices]
    top_scores = top_scores
    top_labels = labels[top_indices]

    # Ground Truth
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    # Если нет аннотаций для этого изображения, пропускаем его
    if len(anns) == 0:
        print("No ground truth annotations found for this image.")
        return

    gt_boxes = [np.array(ann['bbox']).reshape(1, 4) for ann in anns]
    gt_boxes = np.concatenate(gt_boxes)
    gt_boxes[:, 2:] += gt_boxes[:, :2]  # xywh -> xyxy
    gt_classes = [ann['category_id'] for ann in anns]

    # Визуализация
    visualize_image(img_id, top_boxes.cpu().numpy(), top_labels.cpu().numpy(), top_scores.cpu().numpy(), gt_boxes, gt_classes)

# Выбираем одно изображение для визуализации
img_id = imgIds[0]  # Первое изображение из датасета
process_image(img_id)