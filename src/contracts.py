# src/contracts.py
from dataclasses import dataclass, field
from typing import List, Tuple
from enum import Enum

# 动作契约 (Action Contract)
class Action(Enum):
    """定义了所有可能的AI动作，用于决策层与执行层的解耦。"""
    NOOP = 0  # No-Operation, 无操作
    JUMP = 1
    DUCK = 2

# 感知层契约 (Perception Contract)
@dataclass(frozen=True)
class Detection:
    """定义了单个检测到的物体，由感知层输出。"""
    label: str
    box: Tuple[int, int, int, int] # (x1, y1, x2, y2)
    confidence: float

# 世界建模层契约 (World Modeling Contract)
@dataclass(frozen=True)
class Obstacle:
    """定义了单个障碍物的完整状态，由世界模型计算。"""
    label: str
    box: Tuple[int, int, int, int]
    distance: float # 与恐龙的水平距离
    speed: float    # 估算出的水平速度

@dataclass(frozen=True)
class WorldState:
    """定义了某一帧的完整世界状态，是世界模型的核心输出。"""
    is_valid: bool # 状态是否有效（例如，是否检测到恐龙）
    dino_box: Tuple[int, int, int, int]
    # 总是返回一个列表，即使没有障碍物
    obstacles: List[Obstacle] = field(default_factory=list)

    @property
    def closest_obstacle(self) -> Obstacle | None:
        """提供一个方便的接口来获取最近的障碍物。"""
        if not self.obstacles:
            return None
        return min(self.obstacles, key=lambda o: o.distance)