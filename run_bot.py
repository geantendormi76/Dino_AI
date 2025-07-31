# run_bot.py (V15 - 决策链条诊断 & 最终稳定版)
import sys
from pathlib import Path
import cv2
import mss
import time
import numpy as np
import onnxruntime as ort

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.perception.detector import AdvancedDetector
from src.world_modeling.state import GameState
from src.world_modeling.world_model import UKFWorldModel
from src.utils.screen_manager import ScreenManager
from src.controls.agent import GameAgent
from src.state_builder import build_state_vector

def main():
    print("🚀 启动 Dino AI (专家大脑 - 最终决战版)...")

    detector = None
    prev_time = time.time() 

    try:
        detector = AdvancedDetector(
            yolo_model_path="models/detection/dino_detector.onnx",
            classifier_model_path="models/classification/dino_classifier.pth"
        )
        game_state = GameState()
        world_model = UKFWorldModel()
        agent = GameAgent()
        screen_manager = ScreenManager(mss.mss())
    except Exception as e:
        print(f"❌ 错误：初始化核心模块失败: {e}")
        return

    # --- PyTorch Profiler 启用 (诊断结束后请注释或移除) ---
    # detector.enable_profiler(log_dir="runs/classifier_profiler_logs")
    # ----------------------------------------------------

    onnx_model_path = "models/policy/dino_policy.onnx"
    print(f"🧠 正在加载专家大脑: {onnx_model_path}")
    try:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        trt_provider_options = {
            "trt_fp16_enable": True,
            "trt_cuda_graph_enable": False, # 暂时禁用 CUDA Graph，因为未实现 I/O Binding
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(project_root / "models" / "onnx_cache"),
            "trt_max_workspace_size": 2147483648, # 2GB 显存工作区
        }
        
        providers = [
            ("TensorrtExecutionProvider", trt_provider_options),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        ort_session = ort.InferenceSession(
            onnx_model_path,
            sess_options=session_options,
            providers=providers
        )
        
        (project_root / "models" / "onnx_cache").mkdir(parents=True, exist_ok=True)

        print(f"✅ 专家大脑加载成功！使用设备: {ort_session.get_providers()}")
        print("注意：第一次运行可能较慢，TensorRT正在构建优化引擎并缓存。")

    except Exception as e:
        print(f"❌ 错误：加载ONNX模型失败: {e}")
        print("请检查ONNX Runtime GPU是否正确安装，以及TensorRT相关库是否可用。")
        print("错误详情:", e)
        if detector and hasattr(detector, 'disable_profiler') and callable(detector.disable_profiler):
            detector.disable_profiler()
        return
    
    screen_manager.select_roi()
    if screen_manager.roi is None:
        print("🔴 未选择游戏区域，程序退出。")
        if detector and hasattr(detector, 'disable_profiler') and callable(detector.disable_profiler):
            detector.disable_profiler() 
        return

    print("3秒后机器人将开始运行...")
    time.sleep(3)
    
    try:
        while True:
            frame_start_time = time.perf_counter()

            current_time = time.time()
            dt = current_time - prev_time
            if dt == 0: dt = 1/60
            prev_time = current_time 

            capture_start = time.perf_counter()
            img = screen_manager.capture()
            capture_end = time.perf_counter()
            if img is None: break
            
            detection_start = time.perf_counter()
            detections = detector.detect(img, yolo_class_names=['bird', 'cactus', 'dino'])
            detection_end = time.perf_counter()
            
            game_state_update_start = time.perf_counter()
            game_state.update(detections, dt)
            game_state_update_end = time.perf_counter()

            world_model_update_start = time.perf_counter()
            closest_obs = min(game_state.obstacles, key=lambda o: o[0][0]) if game_state.obstacles else None
            world_model.update(closest_obs, dt)
            world_model_update_end = time.perf_counter()
            
            state_build_start = time.perf_counter()
            state_vector = build_state_vector(game_state, world_model)
            state_build_end = time.perf_counter()

            # --- 诊断 GameState, WorldModel, StateVector 的内容 ---
            print(f"DEBUG_DECISION: Dino Box: {game_state.dino_box}")
            print(f"DEBUG_DECISION: Obstacles Count: {len(game_state.obstacles)} | Closest: {closest_obs}")
            pos, speed = world_model.get_state()
            print(f"DEBUG_DECISION: World Model State (Pos, Speed): ({pos:.2f}, {speed:.2f})" if pos is not None else "DEBUG_DECISION: World Model State: None")
            print(f"DEBUG_DECISION: State Vector: {state_vector.round(3) if state_vector is not None else 'None'}")
            
            action_index = 0
            inference_start = time.perf_counter()
            if state_vector is not None:
                input_state_tensor = np.expand_dims(state_vector, axis=0).astype(np.float32)

                input_name = ort_session.get_inputs()[0].name
                output_name = ort_session.get_outputs()[0].name 

                raw_action_result = [] 
                try:
                    raw_action_result = ort_session.run([output_name], {input_name: input_state_tensor})
                except Exception as e:
                    print(f"❌ ERROR: 决策模型 (dino_policy.onnx) ORT run 失败！详细错误: {e}")
                    print(f"  输入数据形状: {input_state_tensor.shape}, 类型: {input_state_tensor.dtype}")
                    raw_action_result = [] 

                if raw_action_result and len(raw_action_result) > 0:
                    action_output_tensor = raw_action_result[0]
                    if isinstance(action_output_tensor, np.ndarray) and action_output_tensor.size > 0:
                        action_index = int(action_output_tensor.flatten()[0]) 
                        if action_index not in [0, 1, 2]:
                            print(f"⚠️ 警告：决策模型返回了超出范围的动作索引: {action_index}。默认执行 '无操作'。")
                            action_index = 0
                    else:
                        print(f"⚠️ 警告：决策模型返回非 numpy 数组或空数组。原始结果: {raw_action_result}。默认执行 '无操作'。")
                        action_index = 0
                else:
                    print(f"⚠️ 警告：决策模型未返回任何结果。原始结果: {raw_action_result}。默认执行 '无操作'。")
                    action_index = 0

            inference_end = time.perf_counter()

            # --- 诊断决策动作和执行情况 ---
            print(f"DEBUG_DECISION: Chosen Action Index: {action_index}")

            action_execute_start = time.perf_counter()
            if action_index == 1:
                agent.jump(duration=0.05)
            elif action_index == 2:
                agent.duck()
            action_execute_end = time.perf_counter()

            frame_end_time = time.perf_counter()
            
            print(f"Frame Time: {((frame_end_time - frame_start_time)*1000):.2f}ms | "
                  f"Capture: {((capture_end - capture_start)*1000):.2f}ms | "
                  f"Detect: {((detection_end - detection_start)*1000):.2f}ms | "
                  f"GameState: {((game_state_update_end - game_state_update_start)*1000):.2f}ms | "
                  f"WorldModel: {((world_model_update_end - world_model_update_start)*1000):.2f}ms | "
                  f"StateBuild: {((state_build_end - state_build_start)*1000):.2f}ms | "
                  f"Inference: {((inference_end - inference_start)*1000):.2f}ms | "
                  f"ActionExecute: {((action_execute_end - action_execute_start)*1000):.2f}ms")

            debug_img = img.copy()
            for box, class_name in detections:
                x1, y1, x2, y2 = box
                color_map = {"dino": (0, 255, 0), "cactus": (0, 0, 255), "bird": (255, 0, 0)}
                color = next((c for k, c in color_map.items() if k in class_name), (0, 255, 255))
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(debug_img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            _, speed = world_model.get_state()
            speed_text = f"Speed: {speed or 0:.0f}"
            cv2.putText(debug_img, speed_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.imshow("Dino AI - Expert Brain (Debug View)", debug_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if detector and hasattr(detector, 'disable_profiler') and callable(detector.disable_profiler):
            detector.disable_profiler() 
            
    cv2.destroyAllWindows()
    print("🤖 Dino AI 已停止运行。")
    
if __name__ == "__main__":
    main()