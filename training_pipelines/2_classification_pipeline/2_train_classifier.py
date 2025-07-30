# training_pipelines/2_classification_pipeline/2_train_classifier.py (V1.2 - 最终修正版)
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
import copy # 导入copy模块

# [核心修改] 让脚本能够感知到项目的根目录
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# --- [核心修改] 使用绝对路径定义输入和输出 ---
CONFIG = {
    "dataset_path": project_root / "data" / "classification_data",
    "output_model_dir": project_root / "models" / "classification",
    "output_model_name": "dino_classifier.pth",
    "num_epochs": 15,
    "batch_size": 32,
    "learning_rate": 0.001,
}

def train_classifier():
    """
    使用迁移学习训练一个EfficientNetV2模型来分类恐龙游戏中的障碍物。
    """
    print("开始训练图像分类器...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 确保输出目录存在
    output_dir = Path(CONFIG["output_model_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. 数据预处理和加载
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    dataset_path = Path(CONFIG["dataset_path"])
    print(f"加载数据集: {dataset_path}")
    if not dataset_path.exists() or not any(dataset_path.iterdir()):
        print(f"❌ 错误: 数据集目录 '{dataset_path}' 不存在或为空!")
        print("请先运行 1_generate_dataset.py 来生成数据。")
        return
        
    full_dataset = datasets.ImageFolder(dataset_path)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # --- [核心修正] 使用深拷贝解决Subset共享transform的问题 ---
    val_dataset.dataset = copy.deepcopy(full_dataset)
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']
    # ----------------------------------------------------

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"发现 {num_classes} 个类别: {class_names}")

    # 2. 加载预训练模型并修改最后一层
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
        
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    print("✅ EfficientNetV2 模型加载并修改完成。")

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=CONFIG["learning_rate"])

    # 4. 训练循环
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(CONFIG["num_epochs"]):
        print(f'Epoch {epoch+1}/{CONFIG["num_epochs"]}')
        print('-' * 10)
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                print(f'🎉 新的最佳验证集准确率: {best_acc:.4f}')
    
    time_elapsed = time.time() - since
    print(f'训练完成，耗时 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳验证集准确率: {best_acc:4f}')
    
    # 5. 保存最佳模型
    model.load_state_dict(best_model_wts)
    save_path = output_dir / CONFIG["output_model_name"]
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, save_path)
    print(f"✅ 最佳模型已保存至: {save_path}")

if __name__ == '__main__':
    train_classifier()