# src/world_modeling/world_model.py (V3 - 增强诊断与注释)
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
import time

class UKFWorldModel:
    """
    世界建模层：使用无迹卡尔曼滤波器 (Unscented Kalman Filter, UKF)
    来估计最接近障碍物的状态（位置和速度）。
    这是 L2/L3 级智能体中，从纯粹的“感知”迈向“理解”的关键一步。
    """
    def __init__(self): 
        self.tracker = None

    def _create_tracker(self, initial_x, dt):
        """
        初始化UKF追踪器。
        状态向量 (ukf.x): [position, velocity]
        观测向量 (ukf.z): [position]
        """
        # MerweScaledSigmaPoints 是一种用于UKF的采样点生成策略
        points = MerweScaledSigmaPoints(n=2, alpha=0.1, beta=2.0, kappa=-1)
        
        # dim_x=2: 状态向量维度为2 (位置, 速度)
        # dim_z=1: 观测向量维度为1 (只观测位置)
        ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1, dt=dt, hx=self._hx, fx=self._fx, points=points)

        # 初始状态：位置为首次观测到的位置，速度为0
        ukf.x = np.array([initial_x, 0.0])
        
        # 状态协方差矩阵 (P): 表示状态估计的不确定性。
        # 位置不确定性较小，速度不确定性较大。
        ukf.P = np.diag([100.0, 5000.0]) 
        
        # 测量噪声协方差 (R): 表示我们对观测值的信任程度。值越小，越信任观测。
        ukf.R = np.diag([25.0]) 
        
        # 过程噪声协方差 (Q): 表示我们对状态转移模型(_fx)的信任程度。
        # 值越大，表示我们认为系统动态变化越快，对模型的预测越不信任。
        ukf.Q = np.array([[0.1, 0.0], [0.0, 10.0]])
        
        return ukf

    def _fx(self, x, dt):
        """
        状态转移函数：根据物理模型预测下一时刻的状态。
        这里使用简单的匀速运动模型。
        pos_next = pos + vel * dt
        vel_next = vel
        """
        F = np.array([[1, dt], [0, 1]])
        return F @ x

    def _hx(self, x):
        """
        观测函数：将状态向量映射到观测空间。
        我们只能观测到位置，所以只返回状态向量的第一个元素。
        """
        return np.array([x[0]])

    def update(self, obstacle_measurement, dt):
        """
        用新的观测数据更新世界模型。
        """
        # 如果没有检测到障碍物，重置追踪器。
        # 这是为了防止在障碍物消失后，模型继续错误地预测其位置。
        if obstacle_measurement is None:
            self.tracker = None
            return

        # 我们只关心障碍物的x坐标
        x_pos = obstacle_measurement[0][0]

        # 如果追踪器未初始化，则用当前观测值创建新的追踪器。
        if self.tracker is None:
            self.tracker = self._create_tracker(x_pos, dt)
        
        # UKF标准步骤：预测 -> 更新
        self.tracker.predict(dt=dt)
        self.tracker.update(np.array([x_pos]))
        
    def get_state(self):
        """

        获取当前世界模型的状态（位置和速度）。
        """
        if self.tracker is None:
            return None, None
        
        pos = self.tracker.x[0]
        # 速度取负值，因为障碍物在屏幕上向左移动（x坐标减小），
        # 但我们通常将游戏“速度”理解为一个正值。
        vel = -self.tracker.x[1] 
        return pos, vel