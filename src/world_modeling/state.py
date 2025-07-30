# src/world_modeling/state.py (V5 - 感知时间)

class GameState:
    def __init__(self):
        self.dino_box = None
        self.obstacles = []
        self.dt = 1/30 # 默认值

    def update(self, detections, dt): # 接收dt
        self.dino_box = None
        self.obstacles = []
        self.dt = dt # 保存真实的dt
        
        for box, class_name in detections:
            if class_name == 'dino-player':
                self.dino_box = box
            else:
                self.obstacles.append((box, class_name))