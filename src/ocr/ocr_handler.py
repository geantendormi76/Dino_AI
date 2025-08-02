# src/ocr/ocr_handler.py
from paddleocr import PaddleOCR
import re
import numpy as np

class OCRHandler:
    """
    封装PaddleOCR，专注于识别游戏分数。
    [代码已根据PaddleOCR v3.x官方文档进行最终、最简化修正]
    """
    def __init__(self, lang='en'):
        # [最终修正] 遵循官方文档，使用最简化、最稳健的初始化方式
        # 移除了所有已废弃、冲突或仅命令行可用的参数
        try:
            self.ocr = PaddleOCR(
                lang=lang, 
                # 关键修正：v3.x Python API 使用 'device' 参数指定硬件，而非 'use_gpu'
                device='gpu' 
            )
            print("✅ OCR Handler (PaddleOCR v3.x API - 最简化稳健版) 初始化完成。")
        except Exception as e:
            print(f"❌ 初始化PaddleOCR时发生致命错误: {e}")
            # 将原始错误抛出，以便更清晰地调试（如果还有问题）
            raise e

    def recognize_score(self, image: np.ndarray) -> int | None:
        """
        从给定的图像中识别分数。
        image: 只包含分数区域的图像 (BGR格式)。
        """
        try:
            # 使用新版核心方法 ocr.predict()
            result = self.ocr.predict(image)
            
            # 新版predict返回结果的结构是 List[List[Tuple[List[List[int]], Tuple[str, float]]]]
            if result and result[0]:
                all_texts = [line[1][0] for line in result[0]]
                full_text = "".join(all_texts)
                
                numbers = re.findall(r'\d+', full_text)
                if numbers:
                    return int("".join(numbers))
        except Exception as e:
            print(f"❌ OCR识别时发生错误: {e}")
        
        return None