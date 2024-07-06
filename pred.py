from ultralytics import YOLO
model = YOLO('runs/detect/train15/weights/best.pt')#train22 train21 train15
#'runs/detect/train19/weights/best.pt'
#'runs/detect/train15/weights/best.pt'
import os
for i in os.listdir('jpg2'):
    model.predict('jpg2/'+i, save=True, save_txt=True, augment=True, agnostic_nms=True,retina_masks=True, conf=0.2,
                  iou=0.3)
# augment=True, agnostic_nms=True, iou=0.5, conf=0.3, retina_masks=True