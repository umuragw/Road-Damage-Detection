from ultralytics import YOLO
model = YOLO('runs/detect/train6/weights/best.pt')

# 验证模型
metrics = model.val()  # 无需参数，数据集和设置记忆
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # 包含每个类别的map50-95列表