# training_pipelines/3_policy_pipeline/1_collect_data.py (L3+ 多进程最终修正版)
import sys
from pathlib import Path
import cv2
import mss
import time
import numpy as np
import pynput
import uuid
import os
# --- [核心修正] 统一import风格 ---
import multiprocessing as mp
# 从模块中额外导入需要用于类型提示的类
from multiprocessing import Process, Queue, Event
# ------------------------------------

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
# 感知工作者进程函数 (使用正确的类型提示)
# ==============================================================================
def perception_worker(frame_queue: Queue, state_queue: Queue, reset_event: Event): # type: ignore
    print("🚀 感知工作者进程已启动 (纯计算模式)...")
    
    try:
        fuser = PerceptionFuser()
        state_builder = StateRepresentationBuilder()
    except Exception as e:
        print(f"❌ 感知工作者进程初始化AI模块失败: {e}")
        return

    while True:
        try:
            if reset_event.is_set():
                fuser.reset()
                reset_event.clear()
            
            try:
                full_frame = frame_queue.get_nowait()
            except mp.queues.Empty:
                time.sleep(0.001)
                continue

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
# 主函数 (使用正确的调用方式)
# ==============================================================================
def main():
    print("准备开始采集专家轨迹数据 (最终修正版)...")
    listener = pynput.keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    screen_manager_main = ScreenManager(mss.mss())
    screen_manager_main.select_roi()
    if screen_manager_main.roi is None:
        print("🔴 未选择游戏区域，程序退出。")
        listener.stop()
        return

    # --- 创建进程间通信的队列和事件 (使用mp前缀) ---
    frame_queue = mp.Queue(maxsize=10)
    state_queue = mp.Queue(maxsize=10)
    reset_event = mp.Event()

    # --- 创建并启动独立的感知工作者进程 (直接使用Process类) ---
    perception_process = Process(target=perception_worker, args=(frame_queue, state_queue, reset_event))
    perception_process.daemon = True
    perception_process.start()
    
    print("\n✅ 主进程与感知进程已启动。")
    print("3秒后开始采集... 请点击游戏窗口并准备操作。按 'q' 键停止。")
    time.sleep(3)
    
    print("正在重置追踪器...")
    reset_event.set()
    
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

            captured_frame = screen_manager_main.capture()
            if captured_frame is None: continue
            
            try:
                frame_queue.put_nowait(captured_frame)
            except mp.queues.Full:
                pass

            try:
                latest_state_repr, latest_fused_info, latest_display_frame = state_queue.get_nowait()
            except mp.queues.Empty:
                pass

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

            display_source = latest_display_frame if latest_display_frame is not None else captured_frame
            cv2.imshow("数据采集中 (AI视角)", display_source)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            elapsed_time = time.time() - loop_start_time
            sleep_time = frame_duration - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        print("\n正在停止采集...")
        frame_queue.put(None)
        perception_process.join(timeout=5)
        if perception_process.is_alive():
            print("警告：感知进程超时，强制终止。")
            perception_process.terminate()
        
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
    # Windows下使用多进程需要这个保护
    mp.freeze_support()
    main()