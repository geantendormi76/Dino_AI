# src/controls/agent.py (V4 - 状态感知版 & 增强注释)
import time
from pynput.keyboard import Controller, Key

class GameAgent:
    """
    动作执行层：负责将决策规划层输出的抽象动作（如“跳跃”、“下蹲”），
    转化为对游戏窗口的具体键盘输入。
    """
    def __init__(self):
        # last_action_time: 记录上一次动作的时间戳
        self.last_action_time = 0
        
        # is_busy 和 busy_until_time: 用于实现一个简单的动作冷却或状态锁定。
        # 在L2阶段，这用于防止行为树在一个动作未完成时触发下一个。
        # 在L3阶段，这可以作为一个可选的保护机制，防止决策模型过于“抽搐”。
        # 当前在 run_bot.py 中，我们没有主动调用 check_busy()，
        # 所以这个机制目前是“休眠”的。
        self.is_busy = False 
        self.busy_until_time = 0 
        
        # 初始化 pynput 的键盘控制器
        self.keyboard = Controller()

    def _update_busy_state(self, duration):
        """
        内部方法：更新agent的忙碌状态和时间。
        当一个动作被执行时，将agent标记为“忙碌”，直到指定的持续时间结束。
        """
        self.is_busy = True
        self.last_action_time = time.time()
        self.busy_until_time = self.last_action_time + duration

    def check_busy(self):
        """
        外部检查agent是否忙碌。如果忙碌时间已过，则重置状态。
        """
        if self.is_busy and time.time() > self.busy_until_time:
            self.is_busy = False 
        return self.is_busy

    def jump(self, duration=0.05):
        """
        执行“跳跃”动作。
        模拟按下并释放空格键。

        Args:
            duration (float): 按下空格键的持续时间（秒）。
        """
        # 跳跃动作的总持续时间，可以略长于按键时间，给一个落地缓冲。
        # 这主要用于 _update_busy_state，以定义一个合理的“冷却时间”。
        JUMP_TOTAL_DURATION = 0.45 
        self._update_busy_state(JUMP_TOTAL_DURATION)
        
        print(f"[Action] JUMP triggered.") # 增加诊断打印
        self.keyboard.press(Key.space)
        time.sleep(duration) # 按住空格键一小段时间
        self.keyboard.release(Key.space)
        print(f"[Action] JUMP completed (busy for {JUMP_TOTAL_DURATION}s).")

    def duck(self):
        """
        执行“下蹲”动作。
        模拟按下并释放向下箭头键。
        """
        DUCK_TOTAL_DURATION = 0.4
        self._update_busy_state(DUCK_TOTAL_DURATION)

        print(f"[Action] DUCK triggered.") # 增加诊断打印
        self.keyboard.press(Key.down)
        time.sleep(0.3) # 按住下箭头键
        self.keyboard.release(Key.down)
        print(f"[Action] DUCK completed (busy for {DUCK_TOTAL_DURATION}s).")