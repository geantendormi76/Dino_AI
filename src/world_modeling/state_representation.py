# src/world_modeling/state_representation.py
import numpy as np
import cv2 # 使用OpenCV进行高效的绘制操作

class StateRepresentationBuilder:
    """
    将来自感知融合器的结构化语义信息，转换为决策模型所需的网格化状态表示。
    这种表示方法保留了世界的空间结构。
    """
    # --- 核心配置 ---
    # 定义状态网格的尺寸 (高, 宽)
    GRID_SHAPE = (25, 80) # 高度25个单元，宽度80个单元
    # 定义特征图的通道数
    # 通道0: 恐龙位置
    # 通道1: 小型仙人掌位置
    # 通道2: 大型仙人掌位置
    # 通道3: 鸟的位置
    NUM_CHANNELS = 4
    
    # 游戏区域在原始截图中的像素范围 (x_min, y_min, x_max, y_max)
    # 这个值需要与你的 ScreenManager ROI 范围大致对应
    # 假设游戏区域宽度为800像素，高度为250像素
    GAME_AREA_PIXELS = (0, 0, 800, 250)

    def __init__(self):
        self.grid_h, self.grid_w = self.GRID_SHAPE
        self.game_x_min, self.game_y_min, self.game_x_max, self.game_y_max = self.GAME_AREA_PIXELS
        self.game_w_pixels = self.game_x_max - self.game_x_min
        self.game_h_pixels = self.game_y_max - self.game_y_min
        print("✅ 状态表示构建器 (StateRepresentationBuilder) 初始化完成。")

    def _map_coords_to_grid(self, box: tuple) -> tuple:
        """将像素坐标的边界框映射到网格坐标。"""
        x1, y1, x2, y2 = box
        
        # 计算边界框中心点在游戏区域内的相对位置 (0.0 ~ 1.0)
        center_x_rel = np.clip(((x1 + x2) / 2 - self.game_x_min) / self.game_w_pixels, 0.0, 1.0)
        center_y_rel = np.clip(((y1 + y2) / 2 - self.game_y_min) / self.game_h_pixels, 0.0, 1.0)
        
        # 将相对位置映射到网格坐标
        grid_x = int(center_x_rel * (self.grid_w - 1))
        grid_y = int(center_y_rel * (self.grid_h - 1))
        
        return grid_y, grid_x

    def build(self, fused_info: dict) -> dict | None:
        """
        构建网格化状态。

        Args:
            fused_info (dict): 来自PerceptionFuser的结构化信息。

        Returns:
            dict | None: 包含 "arena_grid" 和 "global_features" 的字典，或在无效状态下返回None。
        """
        # 初始化一个全零的特征图
        arena_grid = np.zeros((self.NUM_CHANNELS, self.grid_h, self.grid_w), dtype=np.float32)

        # 如果没有检测到恐龙，我们认为当前状态无效
        if fused_info["dino_box"] is None:
            return None

        # 1. 绘制恐龙位置
        dino_gy, dino_gx = self._map_coords_to_grid(fused_info["dino_box"])
        arena_grid[0, dino_gy, dino_gx] = 1.0

        # 2. 绘制障碍物位置
        for obstacle in fused_info["obstacles"]:
            obs_gy, obs_gx = self._map_coords_to_grid(obstacle["box"])
            class_name = obstacle["class"]
            if 'cactus-small' in class_name:
                arena_grid[1, obs_gy, obs_gx] = 1.0
            elif 'cactus-large' in class_name:
                arena_grid[2, obs_gy, obs_gx] = 1.0
            elif 'bird' in class_name:
                arena_grid[3, obs_gy, obs_gx] = 1.0
        # 3. 准备全局特征 (目前只有游戏速度)
        # 我们需要对分数进行归一化，这里假设最大分数为99999
        normalized_score = (fused_info["game_score"] or 0) / 99999.0
        global_features = np.array([normalized_score], dtype=np.float32)

        return {
            "arena_grid": arena_grid,
            "global_features": global_features
        }