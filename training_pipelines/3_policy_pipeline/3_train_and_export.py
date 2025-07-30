# training_pipelines/3_policy_pipeline/3_train_and_export.py
import sys
from pathlib import Path
import numpy as np
import torch
from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.datasets import MDPDataset
from d3rlpy.logging import CombineAdapterFactory, FileAdapterFactory, TensorboardAdapterFactory
from d3rlpy.metrics import TDErrorEvaluator, AverageValueEstimationEvaluator

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# --- [核心修改] 更新所有路径 ---
DATASET_PATH = project_root / "data" / "policy_data" / "processed" / "iql_dataset.npz"
LOG_DIR_BASE = project_root / "training_runs" # 将所有训练日志统一存放
EXPERIMENT_NAME = "policy_run"
ONNX_MODEL_PATH = project_root / "models" / "policy" / "dino_policy.onnx"
ONNX_MODEL_PATH.parent.mkdir(exist_ok=True)

TRAIN_STEPS, STEPS_PER_EPOCH, SAVE_INTERVAL_IN_EPOCHS = 200000, 10000, 2

def main():
    print(f"🧠 开始执行“一键式”训练与导出流程...")
    
    # 1. 加载数据集
    try:
        data = np.load(DATASET_PATH)
        dataset = MDPDataset(
            observations=data['observations'], actions=data['actions'],
            rewards=data['rewards'], terminals=data['terminals'],
        )
        print(f"✅ 成功加载并构建数据集: {DATASET_PATH}")
    except FileNotFoundError:
        print(f"❌ 错误：找不到数据集文件 {DATASET_PATH}。请先运行 2_process_data.py。")
        return
        
    # 2. 配置 DiscreteCQL
    cql_config = DiscreteCQLConfig(
        batch_size=256, gamma=0.99, learning_rate=1e-4, alpha=1.0, n_critics=2,
    )

    # 3. 创建实例
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 使用设备: {device}")
    cql = cql_config.create(device=device)

    # 4. 配置日志
    logger = CombineAdapterFactory([
        FileAdapterFactory(root_dir=str(LOG_DIR_BASE)),
        TensorboardAdapterFactory(root_dir=str(LOG_DIR_BASE))
    ])
    
    # 5. 配置评估器
    evaluators = { 'td_error': TDErrorEvaluator(), 'avg_value': AverageValueEstimationEvaluator() }
    
    # 6. 开始训练
    print("🚀 正在启动训练... 你可以通过以下命令启动TensorBoard来实时监控：")
    print(f"   tensorboard --logdir {LOG_DIR_BASE}")
    
    cql.fit(
        dataset, n_steps=TRAIN_STEPS, n_steps_per_epoch=STEPS_PER_EPOCH,
        save_interval=SAVE_INTERVAL_IN_EPOCHS, logger_adapter=logger,
        experiment_name=EXPERIMENT_NAME, with_timestamp=False, evaluators=evaluators,
    )
    
    # 7. 导出最终模型
    final_model_dir = LOG_DIR_BASE / EXPERIMENT_NAME
    print(f"💾 训练完成，正在将最终策略导出为 ONNX 格式到 {ONNX_MODEL_PATH}...")
    cql.save_policy(str(ONNX_MODEL_PATH))
    
    print("\n" + "="*50)
    print("🎉🎉🎉 “一键式”流程执行完毕！你的最优专家大脑已成功铸造！ 🎉🎉🎉")
    print(f"最终产出: {ONNX_MODEL_PATH}")
    print("现在，我们可以满怀信心地去运行最终的 run_bot.py 了！")
    print("="*50)

if __name__ == "__main__":
    main()