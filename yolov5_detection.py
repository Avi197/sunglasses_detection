import os.path

import torch
import cv2

# Model
model = torch.hub.load('/home/phamson/github/yolov5', 'custom',
                       path='/home/phamson/model/yolov5_eyeglass_v2/weights/best.pt',
                       source='local')  # or yolov5n - yolov5x6, custom

# Images
path = '/home/phamson/data/sunglasses/test'
img_path = 'images (4).jpeg'  # or file, Path, PIL, OpenCV, numpy, list
img = os.path.join(path, img_path)
# Inference
results = model(img, size=120)

# Results
results.show()
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
results.pandas().xyxy[0]
# results.crop()

