# src/perception/detector.py (V8 - 保持宽高比版)

from ultralytics import YOLO
import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image # 需要PIL

class AdvancedDetector:
    def __init__(self, yolo_model_path, classifier_model_path, task='detect'):
        # --- 初始化YOLO和分类器模型的代码保持不变... ---
        try:
            if torch.cuda.is_available():
                self.device = 'cuda'
                print("✅ Hardware Check: GPU (CUDA) is available. Inference will run on GPU.")
            else:
                self.device = 'cpu'
                print("⚠️ Hardware Check: GPU (CUDA) not found. Inference will fall back to CPU.")

            self.yolo_model = YOLO(yolo_model_path, task=task)
            print(f"✅ YOLO Detector: 模型 '{yolo_model_path}' 加载成功。")
        except Exception as e:
            print(f"❌ YOLO Detector: 模型加载失败: {e}")
            raise

        try:
            checkpoint = torch.load(classifier_model_path, map_location=self.device)
            self.class_names_classifier = checkpoint['class_names']
            num_classes = len(self.class_names_classifier)
            
            self.classifier_model = models.efficientnet_v2_s()
            num_ftrs = self.classifier_model.classifier[1].in_features
            self.classifier_model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
            
            self.classifier_model.load_state_dict(checkpoint['model_state_dict'])
            self.classifier_model = self.classifier_model.to(self.device)
            self.classifier_model.eval()
            print(f"✅ Classifier: 模型 '{classifier_model_path}' 加载成功。")
            print(f"   -> 分类器类别: {self.class_names_classifier}")
            
            # --- [核心修改] 定义保持宽高比的图像变换 ---
            self.target_size = 224
            self.classifier_transform = transforms.Compose([
                # 自定义一个lambda函数来实现letterbox填充
                transforms.Lambda(lambda img: self.letterbox(img, new_shape=(self.target_size, self.target_size))),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            # ---------------------------------------------
        except Exception as e:
            print(f"❌ Classifier: 模型加载失败: {e}")
            raise
            
        self.last_obstacle_rois = []

    def letterbox(self, im, new_shape=(224, 224), color=(128, 128, 128)):
        # 将OpenCV的BGR图像转为PIL的RGB图像
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        shape = im.size # 当前尺寸 (width, height)
        
        # 计算缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # 计算缩放后的新尺寸
        new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
        dw, dh = (new_shape[0] - new_unpad[0]) / 2, (new_shape[1] - new_unpad[1]) / 2
        
        # 如果需要缩放
        if shape[::-1] != new_unpad:
            im = im.resize(new_unpad, Image.Resampling.LANCZOS)
            
        # 创建一个灰色背景的新画布
        new_im = Image.new('RGB', new_shape, color)
        # 将缩放后的图像粘贴到画布中央
        new_im.paste(im, (int(round(dw)), int(round(dh))))
        return new_im

    # detect方法保持原样，无需修改
    def detect(self, image, yolo_class_names, confidence_threshold=0.4):
        self.last_obstacle_rois = []
        results = self.yolo_model(image, device=self.device, verbose=False, conf=confidence_threshold)
        result = results[0]
        final_detections = []
        obstacle_rois = []
        obstacle_boxes = []

        for box in result.boxes:
            box_coords = box.xyxy[0].cpu().numpy().astype(int)
            yolo_class_id = int(box.cls[0].cpu().numpy())
            yolo_class_name = yolo_class_names[yolo_class_id]

            if yolo_class_name == 'dino':
                final_detections.append((box_coords, 'dino-player'))
                continue
            
            if yolo_class_name in ['cactus', 'bird']:
                x1, y1, x2, y2 = box_coords
                padding = 10 
                h, w, _ = image.shape
                x1_pad = max(0, x1 - padding)
                y1_pad = max(0, y1 - padding)
                x2_pad = min(w, x2 + padding)
                y2_pad = min(h, y2 + padding)
                roi = image[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if roi.size > 0:
                    obstacle_rois.append(roi)
                    obstacle_boxes.append(box_coords)

        if obstacle_rois:
            self.last_obstacle_rois = obstacle_rois
            # 注意：这里的ROI是OpenCV的BGR格式，我们的letterbox函数会处理它
            batch_tensor = torch.stack([self.classifier_transform(roi) for roi in obstacle_rois]).to(self.device)
            
            with torch.no_grad():
                outputs = self.classifier_model(batch_tensor)
                _, preds = torch.max(outputs, 1)
            
            for i, pred_idx in enumerate(preds):
                specific_class_name = self.class_names_classifier[pred_idx.item()]
                final_detections.append((obstacle_boxes[i], specific_class_name))
                
        return final_detections