import gym
from gym import spaces
from model import Actor_Critic
from utils import FloorplanOptimizer
import numpy as np
import torch
import random

# 检测GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FloorplanEnv:
    """将布图优化问题封装为强化学习环境"""
    def __init__(self, areas):
        self.optimizer = FloorplanOptimizer(areas)
        self.action_space = spaces.Discrete(6)  # 6种离散动作
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(len(areas)*3,),  # 每个模块的面积、位置x、位置y
            dtype=np.float32
        )
        self.state_dim = len(areas) * 3  # 保持向后兼容
        
    def reset(self):
        self.optimizer = FloorplanOptimizer(self.optimizer.areas)
        return self._get_state()
    
    def step(self, action, agent):
        # 保存当前状态和评估值
        old_state = self._get_state()
        old_value = self.optimizer.evaluate()
        
        # 执行移动操作
        move_success = False
        while  not move_success :
            move_success = self.optimizer.move_operation(action)
        

        # aspect_valid = move_success and self.optimizer._check_overlap() 
        
        # 计算新状态评估值
        # new_value = self.optimizer.evaluate() if aspect_valid else float('inf')
        new_value = self.optimizer.evaluate()
    
        #new_value = self.optimizer.evaluate() if aspect_valid else old_value
        
        # 执行动作前先让智能体决定是否接受改变
        # accept_prob = agent.get_accept_probability(old_state)
        # #accept = random.random() < (accept_prob and aspect_valid)
        # accept = random.random() < accept_prob
        # accept = aspect_valid and (random.random() < accept_prob)
        accept = new_value < old_value

        if not accept:
            # 拒绝则恢复原状态
            self.optimizer.revert_move()
            reward = 0  # 使用原评估值作为奖励
            # if not move_success:
            #     reward = -0.01
        else:
            reward = (old_value - new_value) / old_value  # 使用新评估值作为奖励
            
        # 添加终止条件：当布局评估值足够小或达到最大步数时终止
        done = new_value < 0.1  # 0.1是目标阈值，可根据实际情况调整
        # return self._get_state(), reward, done, {}
        return self._get_state(), reward, done, {'accepted': accept}
        # return self._get_state(), reward, done, {
        #     'accepted': accept,
        #     'aspect_valid': aspect_valid
        # }

        
    def _get_state(self):
        """获取当前状态表示"""
        state = []
        for i in range(self.optimizer.num_blocks):
            state.extend([
                self.optimizer.areas[i],
                self.optimizer.positions[i][0],
                self.optimizer.positions[i][1]
            ])
        return np.array(state)


def train():
    # 设置所有随机种子
    seed = 42  # 可以设置为任意整数
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 示例芯粒面积
    areas = [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 37, 44, 7, 50]
    
    # 创建环境和Agent
    env = FloorplanEnv(areas)
    agent = Actor_Critic(env)  # GPU支持已在类内部处理
    
    # 训练参数
    episodes = 501
    print_interval = 50
    
    # 训练循环
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            steps += 1
            if steps > 1000:  # 安全限制
                #print(f"达到最大步数限制(1000),终止当前episode")
                break
            # action, log_prob = agent.get_action(state)
            # next_state, reward, done, info = env.step(action, agent)

            # agent.learn(log_prob, state, next_state, reward)
            
            # state = next_state
            # total_reward += reward
            # 在训练循环中修改这部分：
            action, log_prob = agent.get_action(state)
            next_state, reward, done, info = env.step(action, agent)

            # 存储经验并触发批量学习
            agent.buffer.append((log_prob, state, next_state, reward))
            agent.learn()  # 无参数调用

            state = next_state
            total_reward += reward

            
        
        if ep % print_interval == 0:
            print(f"Episode {ep}, Total Reward: {total_reward:.2f}")
            w,h = env.optimizer._calculate_contour()
            l = env.optimizer._calculate_hpwl()
            print(f" Area: {w*h}, Hpwl: {l}")
            print(f" W,H: {w, h}")
            env.optimizer.visualize_2()


if __name__ == "__main__":
    train()
