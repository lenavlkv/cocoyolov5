import os
import xml.etree.ElementTree as ET

def convert_voc_to_tlbr(xml_dir, output_dir, class_names):
    os.makedirs(output_dir, exist_ok=True)

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'):
            continue

        tree = ET.parse(os.path.join(xml_dir, xml_file))
        root = tree.getroot()

        txt_filename = os.path.splitext(xml_file)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)

        with open(txt_path, 'w') as f:
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in class_names:
                    continue

                class_id = class_names.index(class_name)
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)  # TLX
                ymin = float(bbox.find('ymin').text)  # TLy
                xmax = float(bbox.find('xmax').text)  # BRx
                ymax = float(bbox.find('ymax').text)  # BRy

                # Class_ID TLX TLy BRx BRy
                f.write(f"{class_id} {xmin} {ymin} {xmax} {ymax}\n")

CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
convert_voc_to_tlbr(
    xml_dir="VOCdevkit/VOC2012/Annotations",
    output_dir="VOCdevkit/VOC2012/labels_tlbr",
    class_names=CLASS_NAMES
)

