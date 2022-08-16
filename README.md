## Sunglasses vs Eyeglasses detection using Yolov5

There is not a lot of good "sunglasses" detection model, so I created one with the least amount of effort possible using
yolov5

Use an existing yolov3 model to detect eyeglasses </br>
Use normal glass data to create yolo labels with normal glass </br>
Use sunglasses data to create yolo labels with sunglasses </br>
Use those 2 labels to train a new yolo model to detect normal glass and sunglasses </br>


#### Yolov3 eyeglasses

This model detect any kind of eyeglasses

https://github.com/Leotiv-Vibs/sunglasses_detection

#### Normal eyeglass dataset

Use preprocess_meglass_data.py to get normal glasses data

https://github.com/cleardusk/MeGlass

#### Sunglasses dataset

https://github.com/shreyas0906/Selfies-with-sunglasses

### Usage

Run ```glass_detection.py``` to create normal glasses labels and sunglasses labels dataset

Train new dataset with ```eyeglass_data.yaml``` file using yolov5

Using only 3 epochs already give acceptable result

```python train.py --img 120 --batch 16 --epochs 3 --data eyeglasses_data.yaml --weights yolov5s.pt```

### Results

Check /results for more

![normal_glasses](https://github.com/Avi197/sunglasses_detection/blob/master/results/eyeglasses_1.jpg)

![sunglasses](https://github.com/Avi197/sunglasses_detection/blob/master/results/sunglasses_1.jpg)



