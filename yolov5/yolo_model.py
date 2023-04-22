import torch
from IPython.display import Image
import os 
import random
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

random.seed(108)

FILE = Path("detect.py").resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from utils.augmentations import letterbox



#Mettre ce bloc dans le __init__ :
#
#     # Load model
#      device = select_device('')
#      model = DetectMultiBackend(self.weights, device=device, dnn=False, data=self.data, fp16=False)
#      stride, names, pt = model.stride, model.names, model.pt
#      imgsz = check_img_size(self.imgsz, s=stride)  # check image size


class Model:
  def __init__(self, weights, data, imgsz=(640, 640)):
    self.weights = weights  # model.pt path(s)
    self.data = data        # dataset.yaml path
    self.imgsz = imgsz      # inference size (height, width)
    self.max_det = 300     # maximum detections per image
    self.augment = False    # augment inference
    self.agnostic_nms = False  # class-agnostic NMS
    self.classes=None         # filter by class: --class 0, or --class 0 2 3

    self.device = select_device('')
    self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False, data=self.data, fp16=False)
    self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
    self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
  
  def predict(self,
          source=ROOT / '../dataset/images/test',  # file/dir/URL/glob, 0 for webcam
          conf_thres=0.25,  # confidence threshold
          iou_thres=0.45,  # NMS IOU threshold
          ):
      source = str(source)
      is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
      is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
      webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
      preds = []
      images = []

      if is_url and is_file:
          source = check_file(source)  # download

      # Load model
##      device = select_device('')
##      model = DetectMultiBackend(self.weights, device=device, dnn=False, data=self.data, fp16=False)
##      stride, names, pt = model.stride, model.names, model.pt
##      imgsz = check_img_size(self.imgsz, s=stride)  # check image size

      # Dataloader
      if webcam:
          view_img = check_imshow()
          cudnn.benchmark = True  # set True to speed up constant image size inference
          dataset = LoadStreams(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
          bs = len(dataset)  # batch_size
      else:
          dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
          bs = 1  # batch_size
      vid_path, vid_writer = [None] * bs, [None] * bs


      # Run inference
      self.model.warmup(imgsz=(1 if self.pt else bs, 3, *self.imgsz))  # warmup
      dt, seen = [0.0, 0.0, 0.0], 0
      for path, im, im0s, vid_cap, s in dataset:
          t1 = time_sync()
          im = torch.from_numpy(im).to(self.device)
          im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
          im /= 255  # 0 - 255 to 0.0 - 1.0
          if len(im.shape) == 3:
              im = im[None]  # expand for batch dim
          t2 = time_sync()
          dt[0] += t2 - t1

          
          # Inference
          pred = self.model(im, augment=self.augment, visualize=False)
          t3 = time_sync()
          dt[1] += t3 - t2

          # NMS
          pred = non_max_suppression(pred, conf_thres, iou_thres, self.classes, self.agnostic_nms, max_det=1000)
          dt[2] += time_sync() - t3

          # Process predictions
          for i, det in enumerate(pred):  # per image
              
              seen += 1
              if webcam:  # batch_size >= 1
                  p, im0, frame = path[i], im0s[i].copy(), dataset.count
                  s += f'{i}: '
              else:
                  p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

              images.append(im0)

              s += '%gx%g ' % im.shape[2:]  # print string
              gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
              if len(det):
                  # Rescale boxes from img_size to im0 size
                  det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
              preds.append(det[:, :6])


          # Print time (inference-only)
          LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

      
      return images, preds


  def predict_image(self,
          im0,  # image
          conf_thres=0.25,  # confidence threshold
          iou_thres=0.45,  # NMS IOU threshold
          ):     


      # Load model
##      device = select_device('')
##      model = DetectMultiBackend(self.weights, device=device, dnn=False, data=self.data, fp16=False)
##      stride, names, pt = model.stride, model.names, model.pt
##      imgsz = check_img_size(self.imgsz, s=stride)  # check image size

      # Pad image
      self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
      im = letterbox(im0, self.imgsz, stride=self.stride, auto=self.pt)[0]
      im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
      im = np.ascontiguousarray(im)

      # Run inference
      self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup
      dt, seen = [0.0, 0.0, 0.0], 0

      t1 = time_sync()
      im = torch.from_numpy(im).to(self.device)
      im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
      im /= 255  # 0 - 255 to 0.0 - 1.0
      if len(im.shape) == 3:
          im = im[None]  # expand for batch dim
      t2 = time_sync()
      dt[0] += t2 - t1

          
      # Inference
      pred = self.model(im, augment=self.augment, visualize=False)
      t3 = time_sync()
      dt[1] += t3 - t2

      # NMS
      pred = non_max_suppression(pred, conf_thres, iou_thres, self.classes, self.agnostic_nms, max_det=1000)
      dt[2] += time_sync() - t3

      # Process predictions
      for i, det in enumerate(pred):  # per image
          seen += 1

          if len(det):
              # Rescale boxes from img_size to im0 size
              det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

      return im0, det[:, :6]
