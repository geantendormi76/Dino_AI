# training_pipelines/3_policy_pipeline/2_process_data.py
import sys
from pathlib import Path
import numpy as np
import os
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# --- [核心修改] 更新输入和输出目录 ---
RAW_DATA_DIR = project_root / "data" / "policy_data" / "raw"
PROCESSED_DATA_DIR = project_root / "data" / "policy_data" / "processed"
PROCESSED_DATA_DIR.mkdir(exist_ok=True, parents=True)
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "iql_dataset.npz"

SURVIVAL_REWARD, CRASH_PENALTY, EPISODE_TIMEOUT_SECONDS = 0.1, -100.0, 2.0

def process_raw_data():
    print(f"🔍 开始从 {RAW_DATA_DIR} 读取原始数据文件...")
    files = sorted(list(RAW_DATA_DIR.glob("*.npz")), key=os.path.getmtime)
    if not files:
        print(f"❌ 错误：在 {RAW_DATA_DIR} 找不到 .npz 文件。请先运行 1_collect_data.py。")
        return
    print(f"✅ 找到了 {len(files)} 个数据点。")
    
    observations, actions, rewards, terminals, next_observations = [], [], [], [], []
    
    print("🔄 开始处理数据，构建转换序列 (s, a, r, s', d)...")
    for i in tqdm(range(len(files) - 1), desc="Processing Transitions"):
        current_data, next_data = np.load(files[i]), np.load(files[i+1])
        done = (os.path.getmtime(files[i+1]) - os.path.getmtime(files[i])) > EPISODE_TIMEOUT_SECONDS
        
        observations.append(current_data['state'])
        actions.append(np.argmax(current_data['action']))
        rewards.append(CRASH_PENALTY if done else SURVIVAL_REWARD)
        terminals.append(done)
        next_observations.append(next_data['state'])


    if files:
        last_data = np.load(files[-1])
        observations.append(last_data['state'])
        actions.append(np.argmax(last_data['action']))
        rewards.append(CRASH_PENALTY)
        terminals.append(True)
        next_observations.append(np.zeros_like(last_data['state']))
        
    print(f"💾 数据处理完成，正在保存到 {PROCESSED_DATA_PATH}...")
    np.savez(
        PROCESSED_DATA_PATH,
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.uint8),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=np.array(terminals, dtype=np.float32),
        next_observations=np.array(next_observations, dtype=np.float32),
    )
    print("🎉 成功！IQL数据集已准备就绪。")

if __name__ == "__main__":
    process_raw_data()