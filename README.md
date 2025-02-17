# COCO YOLOv5x
info from https://pytorch.org/hub/ultralytics_yolov5/

![2025-02-16_19-00-16](https://github.com/user-attachments/assets/82bfa5a8-0e86-4ed5-b75e-f947a8b31c33)

1) mAP from classic_map_compute.py = 0.6835

2) mAP from map_2predictions_without_resize_thr03.py = 0.6678

-------------------------------------------------------------------------------------------------------
classic_map_compute.py - самый обычный базовый подсчет mAP

results_classic_map.txt - результаты базового подсчета mAP

all_predictions_per_box_1image.py - 1 картинка, вывод всех предсказаний для каждого прямоугольника с логитами, классами и score

results_all_predictions_per_box_1image.txt - результаты для 1 картинки, вывод всех предсказаний для каждого прямоугольника с логитами, классами и score

map_2predictions_without_resize_thr03.py - подсчет mAP, 2 наиболее вероятных предсказания на каждый прямоугольник, без ресайза, порог score > 0.3

results_without_resize_thr03 - результаты подсчета mAP, 2 наиболее вероятных предсказания на каждый прямоугольник, без ресайза, порог score > 0.3
