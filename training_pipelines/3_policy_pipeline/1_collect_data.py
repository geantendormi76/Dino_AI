# training_pipelines/3_policy_pipeline/1_collect_data.py (L3+ 多进程最终优化版)
import sys
from pathlib import Path
import cv2
import mss
import time
import numpy as np
import pynput
import uuid
import os
import multiprocessing as mp

# 修正Python模块搜索路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.perception.fuser import PerceptionFuser
from src.world_modeling.state_representation import StateRepresentationBuilder
from src.utils.screen_manager import ScreenManager

# --- 核心配置 ---
RAW_DATA_DIR = project_root / "data" / "policy_data" / "raw"
RAW_DATA_DIR.mkdir(exist_ok=True, parents=True)

ACTION_MAP = {'noop': 0, 'jump': 1, 'duck': 2}
current_action_str = 'noop'

# --- 键盘监听函数 ---
def on_press(key):
    global current_action_str
    if key == pynput.keyboard.Key.space:
        current_action_str = 'jump'
    elif key == pynput.keyboard.Key.down:
        current_action_str = 'duck'

def on_release(key):
    global current_action_str
    if key in [pynput.keyboard.Key.space, pynput.keyboard.Key.down]:
        current_action_str = 'noop'

# ==============================================================================
# [核心优化] 感知工作者进程函数 (不再负责屏幕捕获)
# ==============================================================================
def perception_worker(frame_queue: mp.Queue, state_queue: mp.Queue):
    """
    这个函数在独立的子进程中运行。
    [优化后] 它的职责非常纯粹：只负责纯AI计算。
    它从`frame_queue`获取主进程捕获好的原始截图，进行处理，
    并将结果放入`state_queue`。
    """
    print("🚀 感知工作者进程已启动 (纯计算模式)...")
    
    try:
        fuser = PerceptionFuser()
        state_builder = StateRepresentationBuilder()
    except Exception as e:
        print(f"❌ 感知工作者进程初始化AI模块失败: {e}")
        return

    while True:
        try:
            full_frame = frame_queue.get()
            if full_frame is None:
                break

            fused_info = fuser.fuse(full_frame)
            state_repr = state_builder.build(fused_info)
            
            try:
                state_queue.put_nowait((state_repr, fused_info, full_frame))
            except mp.queues.Full:
                pass

        except Exception as e:
            print(f"感知工作者进程在循环中出错: {e}")
            break
            
    print("🛑 感知工作者进程已正常停止。")

# ==============================================================================
# [核心重构] 主函数
# ==============================================================================
def main():
    print("准备开始采集专家轨迹数据 (最终优化版)...")
    listener = pynput.keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # 主进程负责所有屏幕交互：选择ROI和后续的捕获
    screen_manager_main = ScreenManager(mss.mss())
    screen_manager_main.select_roi()
    if screen_manager_main.roi is None:
        print("🔴 未选择游戏区域，程序退出。")
        listener.stop()
        return

    # --- 创建进程间通信的队列 ---
    frame_queue = mp.Queue(maxsize=1)
    state_queue = mp.Queue(maxsize=1)

    # --- 创建并启动独立的感知工作者进程 (不再传递ROI) ---
    perception_process = mp.Process(target=perception_worker, args=(frame_queue, state_queue))
    perception_process.daemon = True
    perception_process.start()
    
    print("\n✅ 主进程与感知进程已启动。")
    print("3秒后开始采集... 请点击游戏窗口并准备操作。按 'q' 键停止。")
    time.sleep(3)
    
    trajectory = {
        'arena_grids': [],
        'global_features': [],
        'actions': [],
        'rewards': [],
        'timesteps': []
    }
    
    last_score = 0
    start_time = time.time()
    
    TARGET_FPS = 30
    frame_duration = 1 / TARGET_FPS
    
    latest_state_repr = None
    latest_fused_info = None
    latest_display_frame = None

    try:
        while True:
            loop_start_time = time.time()

            # 1. 主进程：快速、低延迟地捕获屏幕
            captured_frame = screen_manager_main.capture()
            if captured_frame is None: continue
            
            # 2. 主进程：将【已经捕获好的截图】任务（非阻塞）放入队列，交给感知进程处理
            try:
                # 为了避免拷贝大数组的开销，可以考虑使用共享内存，但这里为了简单直接传递
                frame_queue.put_nowait(captured_frame)
            except mp.queues.Full:
                pass

            # 3. 主进程：从状态队列获取最新的处理结果（非阻塞）
            try:
                latest_state_repr, latest_fused_info, latest_display_frame = state_queue.get_nowait()
            except mp.queues.Empty:
                pass

            # 4. 主进程：使用最新可用的状态信息进行数据记录
            if latest_state_repr is not None and latest_fused_info is not None:
                action = ACTION_MAP[current_action_str]
                current_score = latest_fused_info.get('game_score') or last_score
                reward = current_score - last_score
                last_score = current_score
                timestep = int(time.time() - start_time)
                
                trajectory['arena_grids'].append(latest_state_repr['arena_grid'])
                trajectory['global_features'].append(latest_state_repr['global_features'])
                trajectory['actions'].append(action)
                trajectory['rewards'].append(reward)
                trajectory['timesteps'].append(timestep)

            # 5. 可视化
            display_source = latest_display_frame if latest_display_frame is not None else captured_frame
            cv2.imshow("数据采集中 (AI视角)", display_source)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # 6. 主循环帧率控制
            elapsed_time = time.time() - loop_start_time
            sleep_time = frame_duration - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        # --- 优雅地关闭和清理 ---
        print("\n正在停止采集...")
        frame_queue.put(None)
        perception_process.join(timeout=5)
        if perception_process.is_alive():
            print("警告：感知进程超时，强制终止。")
            perception_process.terminate()
        
        # 保存轨迹数据
        if len(trajectory['actions']) > 10:
            filename = RAW_DATA_DIR / f"trajectory_{uuid.uuid4()}.npz"
            print(f"💾 正在保存轨迹数据到: {filename}")
            np.savez_compressed(
                filename,
                arena_grids=np.array(trajectory['arena_grids'], dtype=np.float32),
                global_features=np.array(trajectory['global_features'], dtype=np.float32),
                actions=np.array(trajectory['actions'], dtype=np.uint8),
                rewards=np.array(trajectory['rewards'], dtype=np.float32),
                timesteps=np.array(trajectory['timesteps'], dtype=np.uint32)
            )
            print("✅ 保存成功！")
        else:
            print("\n🔴 记录的轨迹过短，不进行保存。")

        listener.stop()
        cv2.destroyAllWindows()
        print("数据采集完成！")

if __name__ == "__main__":
    mp.freeze_support()
    main()