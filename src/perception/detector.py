# src/perception/detector.py (V16 - 预处理与后处理黄金标准版)

import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
from torch.profiler import profile, schedule, tensorboard_trace_handler
import onnxruntime as ort 
from pathlib import Path 

# --- 导入 Ultralytics 官方辅助函数 ---
from ultralytics.utils.ops import non_max_suppression, scale_boxes

# --- 辅助函数：Letterbox 预处理 (来自研究报告，精确复现) ---
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width] (H, W)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape) # new_shape is (W, H)

    # Scale ratio (new / old)
    # new_shape[0] is target width, shape[1] is current width
    # new_shape[1] is target height, shape[0] is current height
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    # new_unpad is (width, height)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    # dw, dh are padding amounts for width and height
    # new_shape[0] is target width, new_unpad[0] is current unpadded width
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  
    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif not scaleup: # Added from official source, ensures padding is non-negative if not scaling up
        dw, dh = max(0, dw), max(0, dh)

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize (shape[::-1] is (width, height))
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR) # new_unpad is (width, height)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh) # Return img, ratio (single float r), and pad (tuple (dw, dh))

# --- 辅助函数：图像预处理 (整合 Letterbox, RGB转换，归一化，CHW转换) (来自研究报告) ---
def preprocess_image_for_yolo(original_image, new_shape=(640, 640)):
    # original_image is expected to be BGR from OpenCV imread
    # 1. BGR 到 RGB 转换 (如果使用 OpenCV 读取图像)
    img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # 2. 应用 letterbox 调整大小和填充
    # new_shape=(width, height) for letterbox
    img_padded, ratio, pad = letterbox(img_rgb, new_shape=new_shape, auto=False, scaleup=True) 
    
    # 3. 通道转置 (HWC 到 CHW) (高, 宽, 通道 -> 通道, 高, 宽)
    img_chw = img_padded.transpose((2, 0, 1))

    # 4. 归一化像素值从 0-255 到 0-1
    img_normalized = np.ascontiguousarray(img_chw, dtype=np.float32) # Ensure contiguous memory
    img_normalized /= 255.0  # Normalize to 0-1

    # 5. 添加批次维度 (1, C, H, W)
    img_final = np.expand_dims(img_normalized, 0)

    return img_final, ratio, pad # Return processed image, scale ratio, and padding

class AdvancedDetector:
    def __init__(self, yolo_model_path, classifier_model_path):
        try:
            if torch.cuda.is_available():
                self.device = 'cuda'
                print("✅ Hardware Check: GPU (CUDA) is available. Inference will run on GPU.")
            else:
                self.device = 'cpu'
                print("⚠️ Hardware Check: GPU (CUDA) not found. Inference will fall back to CPU.")

            print(f"🧠 正在加载 YOLO Detector 模型: {yolo_model_path}")
            
            yolo_session_options = ort.SessionOptions()
            yolo_session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            yolo_trt_provider_options = {
                "trt_fp16_enable": True,
                "trt_cuda_graph_enable": False, # 暂时禁用 CUDA Graph，因为未实现 I/O Binding
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(Path(yolo_model_path).parent / "onnx_cache"),
                "trt_max_workspace_size": 1073741824, # 1GB 显存工作区
            }

            yolo_providers = [
                ("TensorrtExecutionProvider", yolo_trt_provider_options),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            
            self.yolo_ort_session = ort.InferenceSession(
                yolo_model_path,
                sess_options=yolo_session_options,
                providers=yolo_providers
            )
            (Path(yolo_model_path).parent / "onnx_cache").mkdir(parents=True, exist_ok=True)
            print(f"✅ YOLO Detector ONNX Runtime Session 加载成功。使用设备: {self.yolo_ort_session.get_providers()}")
            print("注意：YOLO Detector第一次运行可能较慢，TensorRT正在构建优化引擎并缓存。")
            
        except Exception as e:
            print(f"❌ YOLO Detector: 模型加载失败。错误详情: {e}") 
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
            
            self.target_size = 224
            # --- [核心修改 1] 修正 classifier_transform 中 letterbox 的引用 ---
            # letterbox 现在是文件顶部的独立函数，而不是类方法
            self.classifier_transform = transforms.Compose([
                # 直接调用文件顶部的 letterbox 函数，而不是 self.letterbox
                transforms.Lambda(lambda img: letterbox(img, new_shape=(self.target_size, self.target_size), auto=False, scaleup=False)[0]), # letterbox返回(img, ratio, pad)，我们需要img
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            # --------------------------------------------------------------------
        except Exception as e:
            print(f"❌ Classifier: 模型加载失败: {e}")
            raise
            
        self.last_obstacle_rois = []
        self.profiler = None

    def enable_profiler(self, log_dir="runs/profiler_logs"):
        """在需要时手动调用以启用profiler"""
        if self.profiler is None:
            sch = schedule(wait=20, warmup=10, active=10, repeat=1)
            self.profiler = profile(
                schedule=sch,
                on_trace_ready=tensorboard_trace_handler(log_dir),
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True 
            )
            self.profiler.start()
            print(f"PyTorch Profiler started, logs will be saved to {log_dir}")

    def disable_profiler(self):
        """在诊断结束后手动调用以禁用profiler"""
        if self.profiler is not None:
            self.profiler.stop()
            print(self.profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10)) 
            print("PyTorch Profiler stopped.")
            self.profiler = None
            
    def _postprocess_yolo8_output(self, output, original_img_shape, ratio, pad, 
                                  conf_thres=0.01, iou_thres=0.45, yolo_class_names=None): # 默认置信度改为 0.01
        """
        后处理YOLOv8 ONNX (1,7,8400) 原始输出
        output: (1,7,8400) ONNX原始输出 (NumPy array)
        original_img_shape: (height, width) 原始图像尺寸
        ratio: Letterbox 预处理中的缩放比例
        pad: (dw, dh) Letterbox 预处理中的填充量
        """
        if yolo_class_names is None:
            yolo_class_names = ['bird', 'cactus', 'dino'] 

        # 1. 转置并移除批次维度：从 (1, 7, 8400) 到 (8400, 7)
        predictions = np.squeeze(output).T 

        # 2. 分离组件和计算置信度 (报告指出：YOLOv8输出已包含最终置信度，无需额外obj_conf * class_score)
        boxes_xywh_raw = predictions[:, :4]  # x, y, w, h (归一化，相对 640x640 letterbox图像)
        class_scores_raw = predictions[:, 4:] # 类别分数 (report says 4th index onwards are class confs)

        # 找到最高分数的类别ID和其置信度
        max_class_scores = np.max(class_scores_raw, axis=1) # 这就是最终置信度
        class_ids_raw = np.argmax(class_scores_raw, axis=1)

        raw_scores_sorted_idx = np.argsort(max_class_scores)[::-1] 
        top_k_print = min(10, len(raw_scores_sorted_idx)) 
        
        if top_k_print > 0:
            top_k_conf_scores = max_class_scores[raw_scores_sorted_idx[:top_k_print]]
            top_k_class_ids = class_ids_raw[raw_scores_sorted_idx[:top_k_print]]
            
            top_k_class_names = []
            for cid in top_k_class_ids:
                if cid >= 0 and cid < len(yolo_class_names): 
                    top_k_class_names.append(yolo_class_names[cid])
                else:
                    top_k_class_names.append(f"unknown_cls_{cid}")

            print(f"DEBUG: Top-{top_k_print} Raw Pred Confidences: {top_k_conf_scores.round(3)}, Class_IDs: {top_k_class_ids}, Names: {top_k_class_names}")
            if np.max(top_k_conf_scores) < conf_thres:
                print(f"DEBUG: 最高原始置信度 ({np.max(top_k_conf_scores):.3f}) 低于阈值 ({conf_thres:.3f})，可能无检测结果。")
        else:
            print("DEBUG: 无原始预测框。")

        # 5. 根据置信度阈值进行筛选
        keep_indices_conf = max_class_scores > conf_thres
        boxes_xywh_filtered = boxes_xywh_raw[keep_indices_conf]
        max_class_scores_filtered = max_class_scores[keep_indices_conf]
        class_ids_filtered = class_ids_raw[keep_indices_conf]

        if len(boxes_xywh_filtered) == 0:
            print(f"DEBUG: 经置信度 {conf_thres} 筛选后，无有效检测框。") 
            return [], [], []

        # 6. 应用非极大值抑制 (NMS) - 使用 Ultralytics 官方 NMS
        # 将 NumPy 数组转换为 PyTorch Tensor for NMS
        boxes_xyxy_tensor = torch.from_numpy(boxes_xywh_filtered).float() # (M, 4) in xywh format
        scores_tensor = torch.from_numpy(max_class_scores_filtered).float() # (M,)

        # Convert xywh to xyxy for NMS (torchvision.ops.nms or ultralytics.utils.ops.non_max_suppression expects xyxy)
        # Note: ultralytics non_max_suppression expects (batch_size, num_detections, 6) or (batch_size, num_detections, 4+num_classes)
        # The structure is [x1, y1, x2, y2, conf, class_id] after non_max_suppression.
        # So we need to format it before passing to non_max_suppression.
        # Let's rebuild the input to non_max_suppression to match its expectation.

        # Predictions for NMS need to be (N, 6) or (N, 4+num_classes) for NMS
        # Let's make it (N, 6) -> [x1, y1, x2, y2, conf, class_id]
        
        # Convert xywh to xyxy (normalized to 640x640 space) for NMS
        # This is where non_max_suppression expects its boxes
        
        # We need to construct a tensor of shape (N, 6) from our filtered data
        # [x, y, w, h, max_class_score, class_id]
        
        # Convert xywh to xyxy
        x1 = boxes_xywh_filtered[:, 0] - boxes_xywh_filtered[:, 2] / 2
        y1 = boxes_xywh_filtered[:, 1] - boxes_xywh_filtered[:, 3] / 2
        x2 = boxes_xywh_filtered[:, 0] + boxes_xywh_filtered[:, 2] / 2
        y2 = boxes_xywh_filtered[:, 1] + boxes_xywh_filtered[:, 3] / 2

        # Combine into (N, 6) tensor for NMS
        # (x1, y1, x2, y2, conf, class_id)
        nms_input = np.concatenate((x1[:, np.newaxis], y1[:, np.newaxis], 
                                    x2[:, np.newaxis], y2[:, np.newaxis], 
                                    max_class_scores_filtered[:, np.newaxis], 
                                    class_ids_filtered[:, np.newaxis]), axis=1)
        
        nms_input_tensor = torch.from_numpy(nms_input).float()
        
        # NMS 返回一个张量列表，批次中的每张图像对应一个张量。
        # 每个张量包含检测到的对象: [x1, y1, x2, y2, conf, class_id]
        results_from_nms = non_max_suppression(
            nms_input_tensor.unsqueeze(0), # Add batch dimension for NMS input (1, N, 6)
            conf_thres=conf_thres,         # 使用传入的 conf_thres
            iou_thres=iou_thres,           # 使用传入的 iou_thres
            classes=None,                  # 不按特定类别过滤
            agnostic=False,                # 不进行类别无关NMS
            max_det=1000                   # 每张图像的最大检测数量
        )
        
        if not results_from_nms or len(results_from_nms[0]) == 0: # NMS 后无结果
            print(f"DEBUG: 经 NMS (IoU={iou_thres}, Conf={conf_thres}) 后，无有效检测框。") 
            return [], [], []

        # results_from_nms[0] 是一个 PyTorch Tensor，形状为 (num_final_detections, 6)
        final_detections_tensor = results_from_nms[0] 

        # 转换为 NumPy 数组
        final_detections_np = final_detections_tensor.cpu().numpy()

        # 提取 NMS 后的最终框、置信度和类别ID
        final_boxes_xyxy_640 = final_detections_np[:, :4] # 框在 640x640 letterbox 空间
        final_scores_after_nms = final_detections_np[:, 4]
        final_class_ids_after_nms = final_detections_np[:, 5].astype(int)

        # 8. 精确坐标反变换：从 640x640 letterbox 图像映射回原始图像像素
        # 使用 Ultralytics 官方的 scale_boxes 函数
        # original_img_shape is (H, W)
        # new_shape is (W, H)
        # scale_boxes expects (img1_shape, boxes, img0_shape) for scaling boxes from img1 to img0
        # img1_shape: (H, W) of the image where boxes are currently defined (640, 640)
        # img0_shape: (H, W) of the target image (original_image_h, original_image_w)

        # Corrected: scale_boxes expects input boxes to be in (x1, y1, x2, y2) format already
        # And expects img1_shape (H,W) and img0_shape (H,W)
        
        # Convert final_boxes_xyxy_640 (W,H) format to (H,W) for scale_boxes
        # (640,640) is the input_shape to the model which is (W,H) in letterbox function, so it matches.
        
        # Scale boxes back to original image size
        # scale_boxes expects (img1_shape, boxes, img0_shape)
        # img1_shape is (H, W) where boxes are currently defined (640, 640)
        # img0_shape is (H, W) of the target image (original_image_h, original_image_w)
        # The boxes are already x1y1x2y2 format in 640x640 space
        
        # Use new_shape as (H,W) for scale_boxes for clarity
        model_input_hw = (640, 640) # YOLOv8 model input H, W (after letterbox)

        final_boxes_original_scaled = scale_boxes(
            model_input_hw, # Source shape (H,W)
            torch.from_numpy(final_boxes_xyxy_640).float(), # Boxes in source shape
            original_img_shape # Target shape (H,W)
        ).numpy()

        # 确保坐标是整数
        final_boxes_original_scaled = final_boxes_original_scaled.astype(int)

        final_boxes_processed = final_boxes_original_scaled
        final_scores = final_scores_after_nms
        final_class_ids = final_class_ids_after_nms
        
        return final_boxes_processed, final_scores, final_class_ids


    def detect(self, image, yolo_class_names, confidence_threshold=0.01): # 默认置信度改为 0.01 (用于调试)
        self.last_obstacle_rois = []
        final_detections = []
        obstacle_rois = []
        obstacle_boxes = []

        original_image_h, original_image_w, _ = image.shape 
        
        # --- [核心修改 1] 使用 preprocess_image_for_yolo 进行预处理 ---
        processed_img_tensor, ratio_scale, pad_amount = preprocess_image_for_yolo(image, new_shape=(640, 640)) # new_shape=(W,H) for letterbox input_size=640
        
        yolo_input_name = self.yolo_ort_session.get_inputs()[0].name
        
        raw_outputs = [] 
        try:
            # 显式请求输出名称 'output0' (根据脚本输出)
            raw_outputs = self.yolo_ort_session.run(['output0'], {yolo_input_name: processed_img_tensor})
        except Exception as e:
            print(f"❌ ERROR: YOLO ORT session.run failed during inference. Error: {e}")
            return [] 
        
        if not raw_outputs or not isinstance(raw_outputs[0], np.ndarray) or raw_outputs[0].size == 0:
            print(f"⚠️ 警告：YOLO模型原始输出结果为空或非数组。原始输出: {raw_outputs}。跳过检测。")
            return [] 

        # --- [核心修改 2] 调用 _postprocess_yolo8_output 进行后处理 ---
        # 传入原始图像尺寸、letterbox的 ratio 和 pad
        processed_boxes, processed_scores, processed_class_ids = self._postprocess_yolo8_output(
            raw_outputs[0], # yolo_outputs[0] 是 (1, 7, 8400) 的 NumPy 数组
            original_img_shape=(original_image_h, original_image_w), 
            ratio=ratio_scale, 
            pad=pad_amount, # pad_amount 是 (dw, dh)
            conf_thres=confidence_threshold, 
            iou_thres=0.45, 
            yolo_class_names=yolo_class_names
        )
        
        # 从 processed_boxes 中分离出 dino 和障碍物，并进行后续处理
        for i in range(len(processed_boxes)):
            box = processed_boxes[i]
            class_id = processed_class_ids[i]
            
            yolo_name_from_id = yolo_class_names[class_id] if class_id < len(yolo_class_names) else f"unknown_id_{class_id}"
            
            if yolo_name_from_id == 'dino':
                final_detections.append((box, 'dino-player'))
            elif yolo_name_from_id in ['cactus', 'bird']:
                x1, y1, x2, y2 = box
                padding = 10 
                h_img, w_img, _ = image.shape # image 是原始图像
                x1_pad = max(0, x1 - padding)
                y1_pad = max(0, y1 - padding)
                x2_pad = min(w_img, x2 + padding)
                y2_pad = min(h_img, y2 + padding) 

                roi = image[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if roi.size > 0:
                    obstacle_rois.append(roi)
                    obstacle_boxes.append(box) 
        
        if obstacle_rois:
            if self.profiler is not None:
                self.profiler.step()
            
            self.last_obstacle_rois = obstacle_rois
            batch_tensor = torch.stack([self.classifier_transform(roi) for roi in obstacle_rois]).to(self.device)
            
            with torch.no_grad():
                outputs = self.classifier_model(batch_tensor)
                _, preds = torch.max(outputs, 1)
            
            for i, pred_idx in enumerate(preds):
                specific_class_name = self.class_names_classifier[pred_idx.item()]
                final_detections.append((obstacle_boxes[i], specific_class_name))
                
        return final_detections