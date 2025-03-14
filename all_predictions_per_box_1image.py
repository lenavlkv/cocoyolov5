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

annFile = 'annotations/instances_train2017.json'
coco = COCO(annFile)

# Список изображений
imgIds = coco.getImgIds()

img_id = imgIds[0]
img = coco.loadImgs(img_id)[0]
#img_path = f'val2017/{img["file_name"]}'
img_path = 'val2017/000000000285.jpg'
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
    # we are going to register a PyTorch hook on the important parts of the YOLO model,
    # then reverse engineer the outputs to get boxes and logits
    # first, we have to register the hooks to the model *before running inference*
    # then, when inference is run, the hooks will save the inputs/outputs of their respective modules
    #model = YOLO(model_path)
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

    # save and return these for later
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


def plot_image(img_path, results, category_mapping=None, suffix='test', show_labels=True, include_legend=True):
    """
    Display the image with bounding boxes and their corresponding class scores.

    Args:
        img_path (str): Path to the image file.
        results (list): List of dictionaries containing bounding box information.
        category_mapping:
        suffix: what to append to the original image name when saving

    Returns:
        None
    """

    img = Image.open(img_path)
    fig, ax = plt.subplots()
    ax.imshow(img)

    for box in results:
        x0, y0, x1, y1 = map(int, box['bbox'])

        box_color = "r"  # red
        tag_color = "k"  # black
        max_score = max(box['activations'])
        max_category_id = box['activations'].index(max_score)
        category_name = max_category_id

        if category_mapping:
            max_category_name = category_mapping.get(max_category_id, "Unknown")
            category_name = max_category_name

        rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            edgecolor=box_color,
            label=f"{max_category_id}: {category_name} ({max_score:.2f})",
            facecolor='none'
        )
        ax.add_patch(rect)

        if show_labels:
            plt.text(
                x0,
                y0 - 50,
                f"{max_category_id} ({max_score:.2f})",
                fontsize="5",
                color=tag_color,
                backgroundcolor=box_color,
            )

    if include_legend:
        ax.legend(fontsize="5")

    plt.axis("off")
    plt.savefig(f'{os.path.basename(img_path).rsplit(".", 1)[0]}_{suffix}.jpg', bbox_inches="tight", dpi=300)


def write_json(results):
    # Create a list to store the predictions data
    predictions = []

    for result in results:
        image_id = os.path.basename(result['image_id'])#.split('.')[0]
        # image_id = result["image_id"]
        #image_id = os.path.basename(img_path).split('.')[0]
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
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (list): Bounding box coordinates [x1, y1, w1, h1].
        box2 (list): Bounding box coordinates [x2, y2, w2, h2].

    Returns:
        float: Intersection over Union (IoU) value.
    """
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
    """
    Applies Non-Maximum Suppression (NMS) to a list of bounding box dictionaries.

    Args:
        boxes (list): List of dictionaries, each containing 'bbox', 'logits', and 'activations'.
        iou_threshold (float, optional): Intersection over Union (IoU) threshold for NMS. Default is 0.7.

    Returns:
        list: List of selected bounding box dictionaries after NMS.
    """
    # Sort boxes by confidence score in descending order
    sorted_boxes = sorted(boxes, key=lambda x: max(x['activations']), reverse=True)
    selected_boxes = []

    # Keep the box with highest confidence and remove overlapping boxes
    delete_idxs = []
    for i, box0 in enumerate(sorted_boxes):
        for j, box1 in enumerate(sorted_boxes):
            if i < j and calculate_iou(box0['bbox'], box1['bbox']) > iou_threshold:
                delete_idxs.append(j)

    # Reverse the order of delete_idxs
    delete_idxs.reverse()

    # now delete by popping them in reverse order
    filtered_boxes = [box for idx, box in enumerate(sorted_boxes) if idx not in delete_idxs]

    return filtered_boxes


def results_predict(img_path, model, hooks, threshold=0.5, iou=0.7, save_image = False, category_mapping = None):
    """
    Run prediction with a YOLO model and apply Non-Maximum Suppression (NMS) to the results.

    Args:
        img_path (str): Path to an image file.
        model (YOLO): YOLO model object.
        hooks (list): List of hooks for the model.
        threshold (float, optional): Confidence threshold for detection. Default is 0.5.
        iou (float, optional): Intersection over Union (IoU) threshold for NMS. Default is 0.7.
        save_image (bool, optional): Whether to save the image with boxes plotted. Default is False.

    Returns:
        list: List of selected bounding box dictionaries after NMS.
    """
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

    # assumes batch size = 1 (i.e. you are just running with one image)
    # if you want to run with many images, throw this in a loop
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

    # for debugging
    # top10 = sorted(boxes, key=lambda x: max(x['activations']), reverse=True)[:10]
    # plot_image(img_path, top10, suffix="before_nms")

    # NMS
    # we can keep the activations and logits around via the YOLOv8 NMS method, but only if we
    # append them as an additional time to the prediction vector. It's a weird hacky way to do it, but
    # it works. We also have to pass in the num classes (nc) parameter to make it work.
    boxes_for_nms = torch.stack([
        torch.tensor([*b['bbox_xywh'], *b['activations'], *b['activations'], *b['logits']]) for b in boxes
    ], dim=1).unsqueeze(0)

    # do the NMS
    nms_results = ops.non_max_suppression(boxes_for_nms, conf_thres=threshold, iou_thres=iou, nc=detect.nc)[0]

    # unpack it and return it
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
    """
    Run prediction with a YOLO model.

    Args:
        input_path (str): Path to an image file or txt file containing paths to image files.
        model (YOLO): YOLO model object.
        hooks (list): List of hooks for the model.
        threshold (float, optional): Confidence threshold for detection. Default is 0.5.
        iou_threshold (float, optional): Intersection over Union (IoU) threshold for NMS. Default is 0.7.
        save_image (bool, optional): Whether to save the image with boxes plotted. Default is False.
        save_json (bool, optional): Whether to save the results in a json file. Default is False.

    Returns:
        list: List of selected bounding box dictionaries for all the images given as input.
    """
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
    threshold = 0.5
    nms_threshold = 0.7

    # load the model
    model, hooks = load_and_prepare_model(model_path)

    # run inference
    results = run_predict(img_path, model, hooks, save_image=True, score_threshold=threshold, iou_threshold=nms_threshold)

    print("Processed", len(results), "boxes")
    print(results[0])
    print(sorted([a for a in zip(classes,results[0]['activations'])], key= lambda w: w[1], reverse=True))
   # top_2 = sorted([(classes[i], prob) for i, prob in enumerate(results[0]['activations'])], key=lambda x: x[1], reverse=True)[:2]
   # print(top_2)

    cv2.imshow("Image with Predictions", img_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()