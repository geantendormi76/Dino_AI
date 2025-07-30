# src/config.py

import torch
import onnxruntime

# --- 动态硬件检测与配置 ---
def get_optimal_execution_providers():
    """
    智能检测可用的ONNX Runtime执行提供者。
    优先使用CUDA，如果不可用则回退到CPU。
    """
    available_providers = onnxruntime.get_available_providers()
    if 'CUDAExecutionProvider' in available_providers:
        print("Hardware Check: GPU (CUDAExecutionProvider) is available. Using GPU for inference.")
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        print("Hardware Check: GPU (CUDA) not available. Falling back to CPUExecutionProvider.")
        return ['CPUExecutionProvider']

# --- 感知层配置 ---
MODEL_PATH = "models/dino_best.onnx"
# 动态获取最佳的执行配置
ORT_EXECUTION_PROVIDERS = get_optimal_execution_providers() 
# 置信度阈值
CONFIDENCE_THRESHOLD = 0.6

# --- 世界建模层配置 ---
GAME_ROI = (0, 0, 800, 600) 

# --- 决策规划层配置 ---
JUMP_TRIGGER_DISTANCE = 150

# --- 动作执行层配置 ---
ACTION_COOLDOWN = 0.1