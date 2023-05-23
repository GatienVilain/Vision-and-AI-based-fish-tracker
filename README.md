# FishTracking (test DeepSort with Yolov8)

FishDetectionAndTracking is the main code executed on google collab, we have modified predict.py in the YOLOv8 folder.
We do not use TrainShadow because it is a CNN model that we created to dissociate fish from shadow in case the detection model was not good enough. We don’t have data on that model, so it hasn’t been trained.
