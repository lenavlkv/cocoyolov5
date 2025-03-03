# COCO YOLOv5x
info from https://pytorch.org/hub/ultralytics_yolov5/

![2025-02-16_19-00-16](https://github.com/user-attachments/assets/82bfa5a8-0e86-4ed5-b75e-f947a8b31c33)

1) mAP from classic_map_compute.py = 0.6835

2) mAP with resize 640 = 0.5958
  
3) mAP from map_2predictions_without_resize_thr03.py = 0.6678

-------------------------------------------------------------------------------------------------------
classic_map_compute.py - самый обычный базовый подсчет mAP

results_classic_map.txt - результаты базового подсчета mAP

all_predictions_per_box_1image.py - 1 картинка, вывод всех предсказаний для каждого прямоугольника с логитами, классами и score

results_all_predictions_per_box_1image.txt - результаты для 1 картинки, вывод всех предсказаний для каждого прямоугольника с логитами, классами и score

map_2predictions_without_resize_thr03.py - подсчет mAP, 2 наиболее вероятных предсказания на каждый прямоугольник, без ресайза, порог score > 0.3

results_without_resize_thr03 - результаты подсчета mAP, 2 наиболее вероятных предсказания на каждый прямоугольник, без ресайза, порог score > 0.3

2predictions_1image.py - выводит 1 изображение, на котором должно быть отмечены прямоугольники ground truth, прямоугольник predicted, и для predicted подпись двух наиболее вероятных классов

results_2predictions_1image.png - результат изображения, на котором отмечены прямоугольники ground truth, прямоугольник predicted, и для predicted подпись двух наиболее вероятных классов

results_2predictions_1image.txt - результат того, что выводится в терминале после отработки 2predictions_1image.py

------------------------------------------------------------------------------------------------------

Пример результата вывода 1 изображения с отмеченными прямоугольниками, с двумя наиболее вероятными классами для каждого предсказания (2predictions_1image.py):


image 1/1 C:\Users\Admin\PycharmProjects\cocoyolov5\val2017\000000041888.jpg: 480x640 3 birds, 35.1ms

Speed: 1.0ms preprocess, 35.1ms inference, 62.2ms postprocess per image at shape (1, 3, 480, 640)

Processed 3 boxes

Prediction 1: [('bird', 0.9147385358810425), ('person', 9.933935507433489e-05)]

Prediction 2: [('bird', 0.8887053728103638), ('sheep', 0.00013157655484974384)]

Prediction 3: [('bird', 0.8445466160774231), ('sheep', 0.00212645111605525)]



![results_2predictions_1image](https://github.com/user-attachments/assets/54f1a936-91a4-4a7b-ae99-8fab16d5889a)


------------------------------------------------------------------------------------------------------
вывод логитов на базе https://gist.github.com/justinkay/8b00a451b1c1cc3bcf210c86ac511f46 
