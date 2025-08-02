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
        """
        对完整的游戏帧进行感知融合。

        Args:
            full_frame (np.ndarray): 完整的游戏窗口截图 (BGR格式)。

        Returns:
            dict: 包含当前帧所有语义信息的结构化字典。
        """
        # 1. 目标检测
        detections = self.detector.detect(full_frame, yolo_class_names=['bird', 'cactus', 'dino'])

        # 2. OCR识别分数
        x1, y1, x2, y2 = self.SCORE_ROI
        score_image = full_frame[y1:y2, x1:x2]
        game_score = self.ocr_handler.recognize_score(score_image)

        # 3. 结果融合
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