import os.path
import cv2
import glob
from tqdm import tqdm

import numpy as np
from pathlib import Path


def eyeglass_detection(img_path):
    img = cv2.imread(img_path)
    # img = cv2.resize(img, fx=1.3, fy=1.3)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
    return boxes


def get_coord(in_path, out_path):
    for file in tqdm(glob.glob(os.path.join(in_path, '*.jpg'))):
        class_num = file.split('_')[1]
        boxes = eyeglass_detection(img_path=file)
        labels = os.path.join(out_path, f'{Path(file).stem}.txt')
        if boxes:
            with open(labels, 'w') as f:
                for box in boxes:
                    fm = f'{class_num} {box[0]} {box[1]} {box[2]} {box[3]}\n'
                    f.write(fm)


if __name__ == '__main__':
    modelWeights = "/home/phamson/model/yolov3_glasses/yolov3_training_last.weights"
    net = cv2.dnn.readNet(modelWeights, "/home/phamson/model/yolov3_glasses/yolov3_testing.cfg")
    classes = ["sunglasses"]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    img_path = '/home/phamson/data/sunglasses/yolo_eyeglass/yolo_eyeglasses/images'
    labels_path = '/home/phamson/data/sunglasses/yolo_eyeglass/yolo_eyeglasses/labels'

    # sunglass = '/home/phamson/data/sunglasses/yolo_eyeglass/sunglasses'
    # sunglass_label = '/home/phamson/data/sunglasses/yolo_eyeglass/sunglasses_labels'
    #
    # norm_glass = '/home/phamson/data/sunglasses/yolo_eyeglass/norm_glasses'
    # norm_glass_label = '/home/phamson/data/sunglasses/yolo_eyeglass/norm_glasses_labels'

    get_coord(img_path, labels_path, 1)
