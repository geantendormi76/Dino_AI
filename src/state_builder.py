# src/state_builder.py (V2 - 黄金标准校验版 & 增强注释)
import numpy as np

def normalize(value, min_val, max_val):
    """
    将一个值归一化到 [0, 1] 区间。
    """
    # 防止除以零
    if (max_val - min_val) == 0: 
        return 0.0
    # 使用 clip 确保结果在 [0, 1] 范围内
    return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)

def build_state_vector(game_state, world_model):
    """
    构建决策模型所需的状态向量。
    该向量的维度和每个维度的含义，必须与训练决策模型时完全一致。

    当前状态向量 (7个维度):
    [
        0: 游戏速度 (归一化),
        1: 最近障碍物与恐龙的距离 (归一化),
        2: 最近障碍物的高度 (归一化),
        3: 最近障碍物的宽度 (归一化),
        4: 第二近障碍物与恐龙的距离 (归一化),
        5: 第二近障碍物的高度 (归一化),
        6: 第二近障碍物的宽度 (归一化)
    ]
    """
    # --- [核心] 定义归一化的最大值，这些值应与训练时保持一致 ---
    MAX_SPEED = 1000  # 游戏最大速度的估计值
    MAX_DIST = 800    # 障碍物最大距离的估计值
    MAX_HEIGHT = 100  # 障碍物最大高度的估计值
    MAX_WIDTH = 100   # 障碍物最大宽度的估计值

    # 从 game_state 和 world_model 中获取原始信息
    dino_box = game_state.dino_box
    obstacles = game_state.obstacles
    _, speed = world_model.get_state()
    
    # 确保速度不为None
    current_speed = speed if speed is not None else 0
    
    # 初始化状态向量，第一个维度是游戏速度
    state = [normalize(current_speed, 0, MAX_SPEED)]
    
    # 如果没有检测到恐龙，则无法计算距离，返回 None 表示状态无效
    if dino_box is None:
        return None

    # 按x坐标对障碍物进行排序，以便我们能找到最近和第二近的
    obstacles.sort(key=lambda o: o[0][0])
    
    # 填充最近的两个障碍物信息
    for i in range(2):
        if i < len(obstacles):
            # 如果存在第 i 个障碍物
            obs_box, _ = obstacles[i]
            
            # 计算障碍物与恐龙右侧边缘的距离
            dist = obs_box[0] - dino_box[2]
            # 计算障碍物的高度和宽度
            height = obs_box[3] - obs_box[1]
            width = obs_box[2] - obs_box[0]
            
            # 将归一化后的特征添加到状态向量
            state.extend([
                normalize(dist, 0, MAX_DIST), 
                normalize(height, 0, MAX_HEIGHT), 
                normalize(width, 0, MAX_WIDTH)
            ])
        else:
            # 如果不存在第 i 个障碍物（例如，只有一个或没有障碍物）
            # 使用默认值填充，表示“没有障碍物”
            # [1.0, 0.0, 0.0] 表示：距离无穷大 (归一化为1), 高度为0, 宽度为0
            state.extend([1.0, 0.0, 0.0])
            
    # 将 state 列表转换为 float32 类型的 NumPy 数组
    return np.array(state, dtype=np.float32)