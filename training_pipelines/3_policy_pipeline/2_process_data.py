# training_pipelines/3_policy_pipeline/2_process_data.py (L3+ 版本)
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# --- 核心配置 ---
RAW_DATA_DIR = project_root / "data" / "policy_data" / "raw"
PROCESSED_DATA_DIR = project_root / "data" / "policy_data" / "processed"
PROCESSED_DATA_DIR.mkdir(exist_ok=True, parents=True)
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "dino_dt_dataset.npz"

def process_raw_data():
    print(f"🔍 开始从 {RAW_DATA_DIR} 读取原始轨迹文件...")
    files = sorted(list(RAW_DATA_DIR.glob("*.npz")))
    if not files:
        print(f"❌ 错误：在 {RAW_DATA_DIR} 找不到 .npz 文件。请先运行 1_collect_data.py。")
        return
    print(f"✅ 找到了 {len(files)} 个轨迹文件。")
    
    all_arena_grids, all_global_features, all_actions, all_rtgs, all_timesteps = [], [], [], [], []
    
    print("🔄 开始处理轨迹，计算 Reward-to-Go (RTG)...")
    for file_path in tqdm(files, desc="处理轨迹中"):
        try:
            data = np.load(file_path)
            rewards = data['rewards']
            
            # --- 核心逻辑: 计算Reward-to-Go ---
            # 从轨迹的最后一帧开始，反向累加奖励
            rtgs = np.zeros_like(rewards, dtype=np.float32)
            current_rtg = 0
            for t in reversed(range(len(rewards))):
                current_rtg += rewards[t]
                rtgs[t] = current_rtg
            
            # 将处理好的数据添加到总列表中
            all_arena_grids.append(data['arena_grids'])
            all_global_features.append(data['global_features'])
            all_actions.append(data['actions'])
            all_rtgs.append(rtgs)
            all_timesteps.append(data['timesteps'])
        except Exception as e:
            print(f"处理文件 {file_path} 失败: {e}")

    # 将所有轨迹的数据拼接成一个大的数组
    all_arena_grids = np.concatenate(all_arena_grids, axis=0)
    all_global_features = np.concatenate(all_global_features, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    all_rtgs = np.concatenate(all_rtgs, axis=0)
    all_timesteps = np.concatenate(all_timesteps, axis=0)

    print(f"💾 数据处理完成，正在保存到 {PROCESSED_DATA_PATH}...")
    np.savez_compressed(
        PROCESSED_DATA_PATH,
        arena_grids=all_arena_grids,
        global_features=all_global_features,
        actions=all_actions,
        rtgs=all_rtgs,
        timesteps=all_timesteps,
    )
    print("🎉 成功！决策Transformer数据集已准备就绪。")
    print(f"总样本数: {len(all_actions)}")

if __name__ == "__main__":
    process_raw_data()