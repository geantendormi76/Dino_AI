# src/utils/screen_manager.py (V4 - 彻底分离版)
import cv2
import mss
import numpy as np
from PIL import ImageGrab
import os # 用于文件操作

class ScreenManager:
    def __init__(self, sct_instance):
        self.roi = None
        self.sct = sct_instance
        self.temp_screenshot_path = "temp_screenshot.png"

    def select_roi(self):
        """
        [核心修改] 彻底分离截图和GUI操作，根除死锁。
        """
        print("\n准备选择区域... 正在截取您的主屏幕...")
        
        # 步骤1：仅截图，并立刻保存到文件
        try:
            full_screenshot = ImageGrab.grab()
            full_screenshot.save(self.temp_screenshot_path)
            print("截图已保存。")
        except Exception as e:
            print(f"❌ 截图失败: {e}")
            self.roi = None
            return

        # 步骤2：从文件加载静态图片，再进行GUI操作
        try:
            img_from_file = cv2.imread(self.temp_screenshot_path)
            if img_from_file is None:
                raise FileNotFoundError("无法读取保存的截图文件。")

            window_name = "请用鼠标拖动选择Dino游戏区域, 然后按 ENTER 或 SPACE 键确认"
            print(f"在弹出的 '{window_name}' 窗口中操作...")
            
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            roi_coords = cv2.selectROI(window_name, img_from_file, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()
            
            # 步骤3：清理临时文件
            os.remove(self.temp_screenshot_path)

            if sum(roi_coords) == 0:
                print("❌ ROI selection cancelled.")
                self.roi = None
                return
                
            x, y, w, h = roi_coords
            self.roi = {"top": y, "left": x, "width": w, "height": h}
            print(f"✅ 游戏区域选择成功: {self.roi}")
        
        except Exception as e:
            print(f"❌ 选择区域时发生错误: {e}")
            self.roi = None
            if os.path.exists(self.temp_screenshot_path):
                os.remove(self.temp_screenshot_path) # 确保清理
            return
            
    def capture(self) -> np.ndarray:
        if not self.roi:
            return None
        
        sct_img = self.sct.grab(self.roi)
        return cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)