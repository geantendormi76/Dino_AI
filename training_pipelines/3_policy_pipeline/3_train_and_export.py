# training_pipelines/3_policy_pipeline/3_train_and_export.py (L3+ 版本)
import sys
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.brain.decision_transformer import DecisionTransformer

# --- 核心配置 ---
DATASET_PATH = project_root / "data" / "policy_data" / "processed" / "dino_dt_dataset.npz"
MODEL_SAVE_PATH = project_root / "models" / "policy"
MODEL_SAVE_PATH.mkdir(exist_ok=True, parents=True)
ONNX_MODEL_PATH = MODEL_SAVE_PATH / "dino_decision_transformer.onnx"

# --- PyTorch 数据集类 ---
class TrajectoryDataset(Dataset):
    def __init__(self, data, context_len):
        self.data = data
        self.context_len = context_len

    def __len__(self):
        return len(self.data['actions']) - self.context_len

    def __getitem__(self, idx):
        end_idx = idx + self.context_len
        return {
            'arena_grids': torch.from_numpy(self.data['arena_grids'][idx:end_idx]),
            'global_features': torch.from_numpy(self.data['global_features'][idx:end_idx]),
            'actions': torch.from_numpy(self.data['actions'][idx:end_idx]),
            'rtgs': torch.from_numpy(self.data['rtgs'][idx:end_idx]).unsqueeze(-1),
            'timesteps': torch.from_numpy(self.data['timesteps'][idx:end_idx]).unsqueeze(-1)
        }

def main():
    print("🧠 开始训练决策Transformer...")
    
    # 1. 模型配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        'state_channels': 2, # 来自 StateRepresentationBuilder
        'action_dim': 1,     # 动作是一个单一的整数 (0, 1, 2)
        'num_actions': 3,    # 动作空间的类别数
        'hidden_size': 128,
        'n_layer': 3,
        'n_head': 4,
        'dropout': 0.1,
        'max_len': 20, # 决策时参考的上下文长度
        'device': device
    }

    # 2. 加载数据集
    try:
        data = np.load(DATASET_PATH)
        dataset = TrajectoryDataset(data, context_len=config['max_len'])
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
        print(f"✅ 成功加载数据集: {DATASET_PATH}")
    except FileNotFoundError:
        print(f"❌ 错误：找不到数据集文件 {DATASET_PATH}。请先运行 2_process_data.py。")
        return
        
    # 3. 初始化模型、优化器和损失函数
    model = DecisionTransformer(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # 4. 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch+1}/{num_epochs} ---")
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc="训练中"):
            optimizer.zero_grad()

            states = {
                'arena_grid': batch['arena_grids'].to(device),
                'global_features': batch['global_features'].to(device)
            }
            # 将离散动作转换为独热编码 (One-Hot)
            actions = nn.functional.one_hot(batch['actions'], num_classes=config['action_dim']).float().to(device)
            rtgs = batch['rtgs'].to(device)
            timesteps = batch['timesteps'].to(device)
            
            # 获取模型预测
            action_preds = model(states, actions, rtgs, timesteps) # (B, T, num_actions)

            # 计算损失
            # 我们只预测下一个动作，所以目标是 action_preds 的输出
            # 真实动作是输入的 actions
            action_targets = batch['actions'].to(device) # (B, T)
            
            # Reshape for CrossEntropyLoss: (B * T, num_actions) and (B * T)
            loss = loss_fn(action_preds.reshape(-1, config['num_actions']), action_targets.reshape(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}")

    # 5. 保存并导出模型
    print(f"💾 训练完成，正在将最终策略导出为 ONNX 格式到 {ONNX_MODEL_PATH}...")
    torch.save(model.state_dict(), str(MODEL_SAVE_PATH / "dino_dt_model.pth"))
    
    # 导出ONNX需要一个假的输入
    dummy_states = {
        'arena_grid': torch.randn(1, config['max_len'], 2, 25, 80).to(device),
        'global_features': torch.randn(1, config['max_len'], 1).to(device)
    }
    dummy_actions = torch.zeros(1, config['max_len'], config['action_dim']).to(device)
    dummy_rtgs = torch.randn(1, config['max_len'], 1).to(device)
    dummy_timesteps = torch.zeros(1, config['max_len'], 1, dtype=torch.long).to(device)
    
    torch.onnx.export(
        model,
        (dummy_states, dummy_actions, dummy_rtgs, dummy_timesteps),
        str(ONNX_MODEL_PATH),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['arena_grids', 'global_features', 'actions', 'rtgs', 'timesteps'],
        output_names=['output_actions'],
    )
    print("\n" + "="*50)
    print("🎉🎉🎉 决策Transformer大脑已成功铸造！ 🎉🎉🎉")
    print(f"最终产出: {ONNX_MODEL_PATH}")
    print("="*50)

if __name__ == "__main__":
    main()