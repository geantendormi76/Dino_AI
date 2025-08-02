# run_bot.py (L3+ 决策Transformer 版本)
import sys
from pathlib import Path
import cv2
import mss
import time
import numpy as np
import onnxruntime as ort
from collections import deque

# 修正Python模块搜索路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# --- 导入新的L3+核心模块 ---
from src.perception.fuser import PerceptionFuser
from src.world_modeling.state_representation import StateRepresentationBuilder
from src.utils.screen_manager import ScreenManager
from src.controls.agent import GameAgent
from src.contracts import Action # 引入动作枚举

def main():
    print("🚀 启动 Dino AI (L3+ 决策Transformer 大脑)...")

    # --- 1. 初始化核心模块 ---
    try:
        fuser = PerceptionFuser()
        state_builder = StateRepresentationBuilder()
        agent = GameAgent()
        screen_manager = ScreenManager(mss.mss())
    except Exception as e:
        print(f"❌ 错误：初始化核心模块失败: {e}")
        return

    # --- 2. 加载新的决策Transformer ONNX模型 ---
    onnx_model_path = "models/policy/dino_decision_transformer.onnx"
    print(f"🧠 正在加载决策Transformer大脑: {onnx_model_path}")
    try:
        # 这里的 provider 配置可以复用你之前成功的TensorRT配置
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ort_session = ort.InferenceSession(onnx_model_path, providers=providers)
        print(f"✅ 决策Transformer大脑加载成功！使用设备: {ort_session.get_providers()}")
    except Exception as e:
        print(f"❌ 错误：加载ONNX模型失败: {e}")
        return

    # --- 3. 初始化上下文序列 (AI的短期记忆) ---
    # 这个长度必须与训练时使用的上下文长度(max_len)完全一致
    CONTEXT_LEN = 20
    state_deque = deque(maxlen=CONTEXT_LEN)
    action_deque = deque(maxlen=CONTEXT_LEN)
    rtg_deque = deque(maxlen=CONTEXT_LEN)
    timestep_deque = deque(maxlen=CONTEXT_LEN)

    # --- 4. 游戏与主循环设置 ---
    screen_manager.select_roi()
    if screen_manager.roi is None:
        print("🔴 未选择游戏区域，程序退出。")
        return

    print("3秒后机器人将开始运行...")
    time.sleep(3)
    
    # 初始目标回报设为一个较高值，例如期望获得1000分
    # 这个值会随着时间递减
    target_return = 1000.0
    start_time = time.time()
    last_action = 0 # 初始动作为 "无操作"

    try:
        while True:
            frame_start_time = time.perf_counter()

            # --- a. 感知与状态表示 ---
            full_frame = screen_manager.capture()
            if full_frame is None: break
            
            fused_info = fuser.fuse(full_frame)
            state_repr = state_builder.build(fused_info)
            
            if state_repr is None:
                # 游戏可能结束或未开始，跳过决策
                cv2.imshow("Dino AI - L3+ Brain (Debug View)", full_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            # --- b. 更新上下文序列 ---
            # 注意：我们用上一帧的动作来填充当前帧的动作输入
            state_deque.append(state_repr)
            action_deque.append(last_action)
            rtg_deque.append(target_return)
            timestep = int(time.time() - start_time)
            timestep_deque.append(timestep)
            
            # 简单地让目标回报随时间衰减
            target_return = max(0, target_return - 0.1)

            # --- c. 准备模型输入 (填充与塑形) ---
            # 如果记忆还未填满，用0进行左填充
            pad_len = CONTEXT_LEN - len(state_deque)
            
            input_grids = np.stack([s['arena_grid'] for s in state_deque])
            input_grids = np.pad(input_grids, ((pad_len, 0), (0, 0), (0, 0), (0, 0)), 'constant')

            # (同样的方法处理其他输入)
            # ... (此处省略了对 global_features, actions, rtgs, timesteps 的填充代码，请务必补全)
            # ... 补全代码 Start
            input_globals = np.stack([s['global_features'] for s in state_deque])
            input_globals = np.pad(input_globals, ((pad_len, 0), (0, 0)), 'constant')

            input_actions = np.array(list(action_deque), dtype=np.int64)
            input_actions = np.pad(input_actions, ((pad_len, 0)), 'constant')
            input_actions = np.eye(3)[input_actions] # 转为 One-Hot

            input_rtgs = np.array(list(rtg_deque), dtype=np.float32).reshape(-1, 1)
            input_rtgs = np.pad(input_rtgs, ((pad_len, 0), (0, 0)), 'constant')
            
            input_timesteps = np.array(list(timestep_deque), dtype=np.int64).reshape(-1, 1)
            input_timesteps = np.pad(input_timesteps, ((pad_len, 0), (0, 0)), 'constant')
            # ... 补全代码 End


            # 添加批次维度
            input_grids = np.expand_dims(input_grids, axis=0).astype(np.float32)
            input_globals = np.expand_dims(input_globals, axis=0).astype(np.float32)
            input_actions = np.expand_dims(input_actions, axis=0).astype(np.float32)
            input_rtgs = np.expand_dims(input_rtgs, axis=0).astype(np.float32)
            input_timesteps = np.expand_dims(input_timesteps, axis=0).astype(np.int64)

            # --- d. 模型推理 ---
            input_dict = {
                'arena_grids': input_grids,
                'global_features': input_globals, # 训练脚本中可能未使用，但最好传入
                'actions': input_actions,
                'rtgs': input_rtgs,
                'timesteps': input_timesteps
            }
            
            # 注意：ONNX模型的输入名必须与导出时完全一致
            # 我们需要检查并调整这里的 key
            onnx_input_names = [inp.name for inp in ort_session.get_inputs()]
            # 假设ONNX输入名为 'states_arena_grid', 'states_global_features', ...
            # 这里需要根据你的ONNX模型进行调整
            onnx_inputs = {
                "arena_grids": input_grids,
                "global_features": input_globals, # <-- [已修正] 启用全局特征
                "actions": input_actions,
                "rtgs": input_rtgs, # <-- [已修正] 键名必须为 'rtgs'
                "timesteps": input_timesteps
            }

            action_logits = ort_session.run(None, onnx_inputs)[0]
            
            # 我们只关心序列中最后一个时间步的动作预测
            action_index = np.argmax(action_logits[0, -1, :])

            # --- e. 执行动作 ---
            if action_index == Action.JUMP.value:
                agent.jump(duration=0.05)
            elif action_index == Action.DUCK.value:
                agent.duck()
            
            # 更新 last_action 用于下一轮循环
            last_action = action_index

            # --- f. 可视化与退出 ---
            frame_end_time = time.perf_counter()
            print(f"Frame Time: {((frame_end_time - frame_start_time)*1000):.2f}ms | Action: {Action(action_index).name}")

            cv2.imshow("Dino AI - L3+ Brain (Debug View)", full_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        print("🤖 Dino AI (L3+) 已停止运行。")

if __name__ == "__main__":
    main()