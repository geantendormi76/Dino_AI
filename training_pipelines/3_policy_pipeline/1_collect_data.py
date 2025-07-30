# training_pipelines/3_policy_pipeline/1_collect_data.py (V1.1 - 最终路径修正版)
import sys
from pathlib import Path
import cv2
import mss
import time
import numpy as np
import pynput
import uuid
import os # 导入os模块

# [核心] 修正Python模块搜索路径，确保无论从哪里运行都能找到src
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.perception.detector import AdvancedDetector
from src.world_modeling.state import GameState
from src.world_modeling.world_model import UKFWorldModel
from src.utils.screen_manager import ScreenManager

# --- [核心修正] 严格对齐最终目录结构 ---
RAW_DATA_DIR = project_root / "data" / "policy_data" / "raw"
RAW_DATA_DIR.mkdir(exist_ok=True, parents=True)


ACTION_MAP = {'nothing': np.array([1, 0, 0], dtype=np.float32), 'jump': np.array([0, 1, 0], dtype=np.float32), 'duck': np.array([0, 0, 1], dtype=np.float32)}
current_action = ACTION_MAP['nothing']
def on_press(key):
    global current_action
    if key == pynput.keyboard.Key.space: current_action = ACTION_MAP['jump']
    elif key == pynput.keyboard.Key.down: current_action = ACTION_MAP['duck']
def on_release(key):
    global current_action
    if key in [pynput.keyboard.Key.space, pynput.keyboard.Key.down]: current_action = ACTION_MAP['nothing']
def normalize(value, min_val, max_val):
    if (max_val - min_val) == 0: return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
def build_state_vector(game_state, world_model):
    dino_box = game_state.dino_box
    obstacles = game_state.obstacles
    _, speed = world_model.get_state()
    MAX_SPEED, MAX_DIST, MAX_HEIGHT, MAX_WIDTH = 1000, 800, 100, 100
    state = [normalize(speed if speed else 0, 0, MAX_SPEED)]
    if dino_box is None: return None
    obstacles.sort(key=lambda o: o[0][0])
    for i in range(2):
        if i < len(obstacles):
            obs_box, _ = obstacles[i]
            dist = obs_box[0] - dino_box[2]
            state.extend([normalize(dist, 0, MAX_DIST), normalize(obs_box[3] - obs_box[1], 0, MAX_HEIGHT), normalize(obs_box[2] - obs_box[0], 0, MAX_WIDTH)])
        else:
            state.extend([1.0, 0.0, 0.0])
    return np.array(state, dtype=np.float32)

def main():
    print("准备开始采集专家数据...")
    listener = pynput.keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    with mss.mss() as sct:
        # --- [核心修正] 严格对齐最终模型路径 ---
        detector = AdvancedDetector(
            yolo_model_path=str(project_root / "models" / "detection" / "dino_detector.onnx"),
            classifier_model_path=str(project_root / "models" / "classification" / "dino_classifier.pth")
        )
        # --------------------------------------
        
        game_state, world_model, screen_manager = GameState(), UKFWorldModel(), ScreenManager(sct)
        screen_manager.select_roi()
        if screen_manager.roi is None: return

        print(f"开始采集... 数据将保存至 {RAW_DATA_DIR}")
        print("按 'q' 键停止。")
        prev_time = time.time()
        while True:
            current_time = time.time()
            dt = current_time - prev_time
            if dt == 0: dt = 1/60
            prev_time = current_time
            
            img = screen_manager.capture()
            if img is None: continue
            
            detections = detector.detect(img, yolo_class_names=['bird', 'cactus', 'dino'])
            game_state.update(detections, dt)
            
            closest_obs = min(game_state.obstacles, key=lambda o: o[0][0]) if game_state.obstacles else None
            world_model.update(closest_obs, dt)
            state_vector = build_state_vector(game_state, world_model)
            
            if state_vector is not None:
                filename = RAW_DATA_DIR / f"{uuid.uuid4()}.npz"
                np.savez(filename, state=state_vector, action=current_action)

            cv2.imshow("Data Collection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    listener.stop()
    cv2.destroyAllWindows()
    print(f"数据采集完成！")

if __name__ == "__main__":
    main()