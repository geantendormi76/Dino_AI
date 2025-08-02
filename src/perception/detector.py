# src/perception/detector.py (最终的、可诊断、健-壮的黄金标准版)

import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
import onnxruntime as ort 
from pathlib import Path 

from ultralytics.utils.ops import non_max_suppression, scale_boxes

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

class AdvancedDetector:

    def __init__(self, yolo_model_path, classifier_model_path):

        try:
            print(f"🧠 正在加载 YOLO Detector ONNX 模型 (底层模式): {yolo_model_path}")
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            trt_provider_options = {
                "trt_fp16_enable": True, "trt_cuda_graph_enable": False,
                "trt_engine_cache_enable": True, "trt_engine_cache_path": str(Path(yolo_model_path).parent / "onnx_cache"),
                "trt_max_workspace_size": 2147483648,
            }
            providers = [("TensorrtExecutionProvider", trt_provider_options), "CUDAExecutionProvider", "CPUExecutionProvider"]
            self.yolo_ort_session = ort.InferenceSession(yolo_model_path, sess_options=session_options, providers=providers)
            (Path(yolo_model_path).parent / "onnx_cache").mkdir(parents=True, exist_ok=True)
            print(f"✅ YOLO ONNX Session (底层模式) 加载成功。使用设备: {self.yolo_ort_session.get_providers()}")
        except Exception as e:
            raise e
        try:
            self.device = 'cuda' if 'CUDAExecutionProvider' in self.yolo_ort_session.get_providers() else 'cpu'
            print(f"🧠 正在加载分类器模型到目标设备: {self.device}")
            checkpoint = torch.load(classifier_model_path, map_location='cpu')
            self.class_names_classifier = checkpoint['class_names']
            self.classifier_model = models.efficientnet_v2_s()
            num_ftrs = self.classifier_model.classifier[1].in_features
            self.classifier_model.classifier[1] = torch.nn.Linear(num_ftrs, len(self.class_names_classifier))
            self.classifier_model.load_state_dict(checkpoint['model_state_dict'])
            self.classifier_model = self.classifier_model.to(self.device)
            self.classifier_model.eval()
            print(f"✅ 分类器模型 '{classifier_model_path}' 加载成功。")
            self.target_size = 224
            self.classifier_transform = transforms.Compose([
                transforms.Lambda(lambda img: Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))),
                transforms.Resize((self.target_size, self.target_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        except Exception as e:
            raise e


    # [核心修正] 将 detect 方法完整替换为这个基于官方示例的新版本
    def detect(self, image, yolo_class_names, confidence_threshold=0.25):
        # 1. 图像预处理 (与官方示例一致)
        img_height, img_width, _ = image.shape
        img_letterboxed, ratio, (dw, dh) = letterbox(image, auto=False)

        blob = cv2.dnn.blobFromImage(img_letterboxed, 1/255.0, (640, 640), swapRB=True)
        
        # 2. 模型推理
        yolo_input_name = self.yolo_ort_session.get_inputs()[0].name
        raw_outputs = self.yolo_ort_session.run(['output0'], {yolo_input_name: blob})[0]
        
        # 3. 手动解析输出 (关键步骤，遵循官方示例逻辑)
        outputs = np.transpose(raw_outputs) # (8400, 5 + num_classes)
        
        boxes = []
        scores = []
        class_ids = []

        for row in outputs:
            # 提取类别分数和最高分数的类别索引
            classes_scores = row[4:]
            _, max_score, _, max_class_loc = cv2.minMaxLoc(classes_scores)
            
            if max_score >= confidence_threshold:
                # 提取边界框信息 (center_x, center_y, width, height)
                cx, cy, w, h = row[:4]
                
                # 计算左上角坐标
                left = int(cx - w/2)
                top = int(cy - h/2)
                
                boxes.append([left, top, int(w), int(h)])
                scores.append(float(max_score))
                class_ids.append(max_class_loc[1])

        # 4. 应用 OpenCV 的 NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, 0.45)
        
        final_detections = []
        obstacle_rois = []
        obstacle_boxes = []

        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                
                # 将 letterbox 空间的坐标转换回原始图像坐标
                x1 = int((box[0] - dw) / ratio)
                y1 = int((box[1] - dh) / ratio)
                x2 = int((box[0] + box[2] - dw) / ratio)
                y2 = int((box[1] + box[3] - dh) / ratio)
                
                scaled_box = np.array([x1, y1, x2, y2])
                
                class_id = class_ids[i]
                if class_id >= len(yolo_class_names):
                    continue
                
                yolo_name = yolo_class_names[class_id]
                
                if yolo_name == 'dino':
                    final_detections.append((scaled_box, 'dino-player'))
                elif yolo_name in ['cactus', 'bird']:
                    roi = image[y1:y2, x1:x2]
                    if roi.size > 0:
                        obstacle_rois.append(roi)
                        obstacle_boxes.append(scaled_box)

        # 5. 分类器处理 (逻辑不变)
        if obstacle_rois:
            batch_tensor = torch.stack([self.classifier_transform(roi) for roi in obstacle_rois]).to(self.device)
            with torch.no_grad():
                outputs = self.classifier_model(batch_tensor)
                _, preds = torch.max(outputs, 1)
            
            for i, pred_idx in enumerate(preds):
                specific_class_name = self.class_names_classifier[pred_idx.item()]
                final_detections.append((obstacle_boxes[i], specific_class_name))
                
        return final_detections