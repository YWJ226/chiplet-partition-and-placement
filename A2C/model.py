import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

# 定义device变量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    '''
    演员Actor网络
    '''
    def __init__(self, action_dim, state_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 300)
        self.fc2 = nn.Linear(300, action_dim)

        self.ln = nn.LayerNorm(300)

    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s).to(device)
        x = self.ln(F.relu(self.fc1(s)))
        out = F.softmax(self.fc2(x), dim=-1)

        return out


class Critic(nn.Module):
    '''
    评论家Critic网络
    '''
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 300)
        self.fc2 = nn.Linear(300, 1)

        self.ln = nn.LayerNorm(300)

    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s).to(device)
        x = self.ln(F.relu(self.fc1(s)))
        out = self.fc2(x)

        return out


class Actor_Critic:
    def __init__(self, env):
        self.gamma = 0.99
        self.lr_a = 3e-4
        self.lr_c = 5e-4

        self.env = env
        self.action_dim = self.env.action_space.n             #获取描述行动的数据维度
        self.state_dim = self.env.observation_space.shape[0]  #获取描述环境的数据维度

        # 初始化网络并移动到GPU
        self.actor = Actor(self.action_dim, self.state_dim)
        self.critic = Critic(self.state_dim)
        # self.accept = Accept(self.state_dim)
        self.actor.to(device)
        self.critic.to(device)
        # self.accept.to(device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
        # self.accept_optim = torch.optim.Adam(self.accept.parameters(), lr=self.lr_a)

        self.loss = nn.MSELoss()

    def get_action(self, s):
        """获取动作，仅在本次调用中临时调整特定动作概率"""
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s).to(device)
        
        # 1. 获取网络输出的原始概率分布（保留计算图）
        original_probs = self.actor(s)
        
        # 2. 创建可调整的概率副本
        adjusted_probs = original_probs.clone()
        
        # 3. 检查是否需要调整概率（宽高比异常时）
        if hasattr(self.env, 'optimizer'):
            aspect_status = self.env.optimizer.check_aspect_ratio()
            
            # W/H > 2: 提高动作6概率（最上方→右下角）
            if aspect_status == 1:
                # 非inplace方式调整概率
                adjusted_probs = torch.cat([
                    adjusted_probs[:6],
                    adjusted_probs[6:7] * 5,  # 动作6概率放大5倍
                    adjusted_probs[7:]
                ])
                # 重新归一化
                adjusted_probs = adjusted_probs / adjusted_probs.sum()
            
            # W/H < 0.5: 提高动作7概率（最右侧→左上角）
            elif aspect_status == 2:
                adjusted_probs = torch.cat([
                    adjusted_probs[:7],
                    adjusted_probs[7:] * 5  # 动作7概率放大5倍
                ])
                adjusted_probs = adjusted_probs / adjusted_probs.sum()
        
        # 4. 从调整后的概率采样动作
        dist = Categorical(adjusted_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # 5. 返回动作和log概率
        # 注意：下次调用get_action会重新计算，自动"恢复"原始概率
        return action.detach().cpu().numpy(), log_prob



    def learn(self, log_prob, s, s_, rew):
        # 确保输入数据在GPU上
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s).to(device)
        if isinstance(s_, np.ndarray):
            s_ = torch.FloatTensor(s_).to(device)
        if isinstance(rew, (float, int)):
            rew = torch.FloatTensor([rew]).to(device)
            
        #使用Critic网络估计状态值
        v = self.critic(s)
        v_ = self.critic(s_)

        critic_loss = self.loss(self.gamma * v_ + rew, v)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        td = self.gamma * v_ + rew - v          #计算TD误差
        loss_actor = -log_prob * td.detach()
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()
