# src/world_modeling/world_model.py (V2 - 动态dt最终修正版)
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints

class UKFWorldModel:
    def __init__(self): 
        self.tracker = None

    # --- 第1个关键修改：函数定义，现在需要3个参数 (self, initial_x, dt) ---
    def _create_tracker(self, initial_x, dt):
        points = MerweScaledSigmaPoints(n=2, alpha=0.1, beta=2.0, kappa=-1)
        
        ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1, dt=dt, hx=self._hx, fx=self._fx, points=points)

        ukf.x = np.array([initial_x, 0.0])
        ukf.P = np.diag([100.0, 5000.0]) 
        ukf.R = np.diag([25.0]) 
        ukf.Q = np.array([[0.1, 0.0], [0.0, 10.0]])
        return ukf

    def _fx(self, x, dt):
        F = np.array([[1, dt], [0, 1]])
        return F @ x

    def _hx(self, x):
        return np.array([x[0]])

    # --- 第2个关键修改：update函数也需要dt ---
    def update(self, obstacle_measurement, dt):
        if obstacle_measurement is None:
            self.tracker = None
            return

        x_pos = obstacle_measurement[0][0]

        if self.tracker is None:
            # --- 第3个关键修改：调用时传入dt ---
            self.tracker = self._create_tracker(x_pos, dt)
        
        self.tracker.predict(dt=dt)
        self.tracker.update(np.array([x_pos]))
        
    def get_state(self):
        if self.tracker is None:
            return None, None
        
        pos = self.tracker.x[0]
        vel = -self.tracker.x[1] 
        return pos, vel