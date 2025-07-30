# src/controls/agent.py (V4 - 状态感知版)
import time
from pynput.keyboard import Controller, Key

class GameAgent:
    def __init__(self):
        self.last_action_time = 0
        self.is_busy = False # 动作是否正在执行中
        self.busy_until_time = 0 # 动作将持续到何时
        self.keyboard = Controller()

    def _update_busy_state(self, duration):
        """更新agent的忙碌状态和时间"""
        self.is_busy = True
        self.last_action_time = time.time()
        self.busy_until_time = self.last_action_time + duration

    def check_busy(self):
        """外部检查agent是否忙碌"""
        if self.is_busy and time.time() > self.busy_until_time:
            self.is_busy = False # 如果忙碌时间已过，则重置状态
        return self.is_busy

    def jump(self, duration=0.05):
        # 跳跃动作的总持续时间，可以略长于按键时间，给一个落地缓冲
        JUMP_TOTAL_DURATION = 0.45 
        self._update_busy_state(JUMP_TOTAL_DURATION)
        
        self.keyboard.press(Key.space)
        time.sleep(duration)
        self.keyboard.release(Key.space)
        print(f"[Action] JUMP started (busy for {JUMP_TOTAL_DURATION}s).")

    def duck(self):
        DUCK_TOTAL_DURATION = 0.4
        self._update_busy_state(DUCK_TOTAL_DURATION)

        self.keyboard.press(Key.down)
        time.sleep(0.3) 
        self.keyboard.release(Key.down)
        print(f"[Action] DUCK started (busy for {DUCK_TOTAL_DURATION}s).")