import sys
from pathlib import Path
from ultralytics import YOLO
import torch

# [核心修改] 让脚本能够感知到项目的根目录
# 这使得无论我们从哪里运行这个脚本，它都能正确找到其他文件
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

def main():
    """
    训练并导出YOLOv8目标检测模型。
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- [核心修改] 使用绝对路径，确保路径的稳健性 ---
    # 定义所有输入文件的路径
    base_model_path = project_root / "_inputs" / "base_models" / "yolo11n.pt"
    data_config_path = project_root / "training_pipelines" / "1_detection_pipeline" / "data.yaml"
    
    # 定义所有输出的根目录
    output_dir = project_root / "training_runs"
    
    print(f"加载基础模型: {base_model_path}")
    print(f"加载数据配置: {data_config_path}")
    print(f"训练产出将保存至: {output_dir}")

    # 1. 加载预训练模型
    model = YOLO(base_model_path)

    # 2. 训练模型
    print("开始目标检测模型训练...")
    model.train(
        data=str(data_config_path), # 确保传入的是字符串
        epochs=50,
        imgsz=640,
        batch=8,
        project=str(output_dir), # [核心修改] 控制输出目录
        name='detection_run'     # 定义本次实验的名称
    )
    print("训练完成。")

    # 3. 导出模型为ONNX
    # 加载训练出的最佳模型
    best_model_path = model.trainer.best
    model = YOLO(best_model_path)
    
    print(f"正在从 {best_model_path} 导出最佳模型为ONNX...")
    # 导出的ONNX文件会自动保存在与 best.pt 相同的目录下
    model.export(format='onnx', opset=12) 
    print("导出完成。")
    print(f"请手动将最终的 .onnx 文件移动到 'models/detection/' 目录下，并重命名为 'dino_detector.onnx'。")


if __name__ == '__main__':
    main()