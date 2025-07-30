# run_bot.py (V11.0 - 黄金标准 ONNX 推理版)
import sys
from pathlib import Path
import cv2
import mss
import time
import numpy as np
import onnxruntime as ort # 导入ONNX运行时

# [核心] 修正Python模块搜索路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# 导入所有需要的模块
from src.perception.detector import AdvancedDetector
from src.world_modeling.state import GameState
from src.world_modeling.world_model import UKFWorldModel
from src.utils.screen_manager import ScreenManager
from src.controls.agent import GameAgent
from tools.collect_data import build_state_vector # 暂时复用，未来可移入src

def main():
    print("🚀 启动Dino AI (专家大脑 - ONNX版)...")

    # --- 1. 加载所有支持模块 ---
    detector = AdvancedDetector(
        yolo_model_path="models/dino_best.onnx",
        classifier_model_path="models/dino_classifier_best.pth"
    )
    game_state = GameState()
    world_model = UKFWorldModel()
    agent = GameAgent()
    screen_manager = ScreenManager(mss.mss())

    # --- 2. 加载训练好的ONNX“专家大脑” ---
    onnx_model_path = "models/dino_cql_policy.onnx"
    print(f"🧠 正在加载专家大脑: {onnx_model_path}")
    try:
        # 根据文档，创建ONNX推理会话
        ort_session = ort.InferenceSession(
            onnx_model_path, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        print(f"✅ 专家大脑加载成功！使用设备: {ort_session.get_providers()}")
    except Exception as e:
        print(f"❌ 错误：加载ONNX模型失败: {e}")
        print("请确保已完成训练，并且 'models/dino_cql_policy.onnx' 文件存在。")
        return
    
    screen_manager.select_roi()
    if screen_manager.roi is None: return

    print("3秒后机器人将开始运行...")
    time.sleep(3)
    
    prev_time = time.time()
    while True:
        # 如果Agent正忙于执行一个长动作，就跳过所有逻辑，让它完成
        if agent.check_busy():
            time.sleep(0.01)
            continue

        current_time = time.time()
        dt = current_time - prev_time
        if dt == 0: dt = 1/60
        prev_time = current_time
        
        # --- 感知 -> 建模 -> 决策 -> 执行 的最终流程 ---
        # 1. 感知
        img = screen_manager.capture()
        if img is None: continue
        detections = detector.detect(img, yolo_class_names=['bird', 'cactus', 'dino'])
        
        # 2. 世界建模
        game_state.update(detections, dt)
        closest_obs = min(game_state.obstacles, key=lambda o: o[0][0]) if game_state.obstacles else None
        world_model.update(closest_obs, dt)
        state_vector = build_state_vector(game_state, world_model)
        
        # 3. 决策 (使用ONNX模型)
        if state_vector is not None:
            # 准备输入：根据文档，输入需要一个批次维度
            input_state = np.expand_dims(state_vector, axis=0)
            
            # ONNX模型的输入名通常是 'input_0' 或 'observation'，具体取决于导出时的设置
            # 我们可以通过 ort_session.get_inputs()[0].name 动态获取
            input_name = ort_session.get_inputs()[0].name
            
            # 推理
            action_result = ort_session.run(None, {input_name: input_state})[0]
            action_index = action_result[0] # 结果通常在第一个数组的第一个元素

            # 4. 执行
            if action_index == 1: # Jump
                agent.jump(duration=0.05)
            elif action_index == 2: # Duck
                agent.duck()
            # action_index == 0 (Do Nothing) 则不执行任何操作

        cv2.imshow("Dino AI - Expert Brain ONNX", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()