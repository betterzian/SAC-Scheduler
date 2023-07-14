import numpy as np
import math
__NUM__ = 5
__TIMEBLOCK__ = 10

class ENV:

    """
    
    """

    def __init__(self, **params):
        self.__max_resource__ = np.zeros(shape=(__NUM__, __TIMEBLOCK__), dtype=float)
        self.__max_resource__ += 100
        self.__max_resource_sum__ = self.__max_resource__.sum()
        # state 为当前 服务器 资源使用量
        # state[0] 为当前需要调度的任务资源需求量
        self.state = np.random.uniform(1, 10, (__NUM__+1,__TIMEBLOCK__))
        self.state[0] = np.random.uniform(0.1, 15, (1,__TIMEBLOCK__))
        self.fail_count = 0
        self.action = None
        self.reward = None
        self.terminate = False

    def reset(self):
        """
        重置整个环境
        Returns:
            self.state
        """

        self.state = np.random.uniform(1, 10, (__NUM__+1,__TIMEBLOCK__))
        self.state[0] = np.random.uniform(0.1, 15, (1,__TIMEBLOCK__))
        self.fail_count = 0
        self.action = None
        self.reward = None
        self.terminate = False

        return self.state

    def step(self, action: np.ndarray):
        """
        Returns:
            self.state, self.reward, self.terminate
        """
        # 得到函数实例部署调整动作
        self.action = action
        # 任务卸载，计算奖励
        self.reward,is_valid= self.get_reward()
        if is_valid:
            self.state[0] = np.random.uniform(0.1, 15, (1,__TIMEBLOCK__))
        if  self.fail_count > 100:
            self.terminate = True
        return self.state, self.reward, self.terminate

    def get_reward(self):
        if self.action == 0: #动作为0表示跳过此调度任务，不推荐，但后期为了能放下更多的任务，可以有此操作
            self.fail_count += 1
            return -500 ,True
        else:
            self.state[self.action] += self.state[0]
            if np.any(self.state[self.action] > self.__max_resource__[self.action]):
                self.state[self.action] -= self.state[0]
                self.fail_count += 1
                return -500 ,False
            else:
                temp = self.cal_reward()
                return 100 * temp,True
    
    def cal_reward(self):
        x1 = 1 - self.state[self.action].sum() *1.0 / self.__max_resource__[self.action].sum()
        x2 = 1 - (self.state[self.action].sum() - self.state[0].sum() )*1.0 / self.__max_resource__[self.action].sum()
        x1 = - math.log2(x1)
        x2 = - math.log2(x2)
        return x1 - x2
        
    def get_state_dim(self):
        return self.state.size
    
    def get_action_dim(self):
        return __NUM__
    
    def get_usage(self):
        x = (self.state.sum() -self.state[0].sum()) *1.0 / self.__max_resource__.sum()
        return x