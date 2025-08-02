# src/perception/fuser.py
import numpy as np
from src.perception.detector import AdvancedDetector
from src.ocr.ocr_handler import OCRHandler

class PerceptionFuser:
    """
    融合YOLOv8目标检测和OCR文本识别的结果，
    输出一个结构化的、包含游戏世界完整语义信息的字典。
    """
    # 注意：这个ROI是相对于整个游戏窗口截图的，你需要根据你的屏幕分辨率进行调整
    # (x1, y1, x2, y2)
    SCORE_ROI = (550, 20, 620, 40) # 这是一个示例值，你需要精确测量

    def __init__(self):
        # 初始化YOLOv8检测器（复用你已有的）
        self.detector = AdvancedDetector(
            yolo_model_path="models/detection/dino_detector.onnx",
            classifier_model_path="models/classification/dino_classifier.pth"
        )
        # 初始化OCR处理器
        self.ocr_handler = OCRHandler()
        print("✅ 感知融合器 (PerceptionFuser) 初始化完成。")

    def fuse(self, full_frame: np.ndarray) -> dict:
        # [诊断性修改] 根据报告建议，将阈值降至极低水平以捕获任何可能的信号
        confidence_threshold = 0.01  # 保持这个极低的阈值用于诊断
        
        detections = self.detector.detect(
            full_frame, 
            yolo_class_names=['bird', 'cactus', 'dino'],
            confidence_threshold=confidence_threshold
        )
        
        # [诊断日志] 确认从检测器收到的检测数量
        print(f"[Fuser DEBUG] 从检测器收到 {len(detections) if detections is not None else 0} 个检测结果。")

        # 2. OCR识别分数 (逻辑不变)
        x1, y1, x2, y2 = self.SCORE_ROI
        score_image = full_frame[y1:y2, x1:x2]
        game_score = self.ocr_handler.recognize_score(score_image)

        # 3. 结果融合 (逻辑不变)
        fused_info = {
            "dino_box": None,
            "obstacles": [],
            "game_score": game_score
        }

        for box, class_name in detections:
            if "dino" in class_name:
                fused_info["dino_box"] = box
            else:
                fused_info["obstacles"].append({
                    "box": box,
                    "class": class_name
                })
        
        return fused_info
    
    def reset(self):
        """重置内部所有有状态的模块，如追踪器"""
        if hasattr(self.detector, 'tracker'):
            self.detector.tracker.reset()
        print("PerceptionFuser已重置。")