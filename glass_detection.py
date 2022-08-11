import cv2
import numpy as np
import glob
import random


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


if __name__ == '__main__':
    modelWeights = "/home/phamson/data/yolov3_training_last.weights"
    net = cv2.dnn.readNet(modelWeights, "yolov3_testing.cfg")
    classes = ["sunglasses"]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # img_path = '/home/phamson/data/sunglasses/glass.jpg'

    folder_path = ''
    # img = temp(img_path)
    # cv2.imshow("Image", img)
    # key = cv2.waitKey(0)
    # cv2.destroyAllWindows()
