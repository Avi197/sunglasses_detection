sunglasses detection using Yolov5

There is not a lot of good "sunglasses" detection model, so I created one with the least amount of effort possible using
yolov5

Use an existing yolov3 model to detect eyeglasses
Use normal glass data to create yolo labels with normal glass
Use sunglasses data to create yolo labels with sunglasses
Use those 2 labels to train a new yolo model to detect normal glass and sunglasses


#### Yolov3 eyeglasses

This model detect any kind of eyeglasses

https://github.com/Leotiv-Vibs/sunglasses_detection

#### Normal eyeglass dataset

Use preprocess_meglass_data.py to get normal glasses data

https://github.com/cleardusk/MeGlass

#### Sunglasses dataset

https://github.com/shreyas0906/Selfies-with-sunglasses

### Usage

Run glass_detection.py to create normal glasses labels and sunglasses labels dataset

Train new dataset with eyeglass_data.yaml file using yolov5

