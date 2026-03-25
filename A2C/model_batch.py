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

# class Accept(nn.Module):
#     '''
#     接受决策网络
#     '''
#     def __init__(self, state_dim):
#         super(Accept, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 300)
#         self.fc2 = nn.Linear(300, 1)
#         self.ln = nn.LayerNorm(300)

#     def forward(self, s):
#         if isinstance(s, np.ndarray):
#             s = torch.FloatTensor(s).to(device)
#         x = self.ln(F.relu(self.fc1(s)))
#         out = torch.sigmoid(self.fc2(x))
#         return out


class Actor_Critic:
    def __init__(self, env):
        self.gamma = 0.99
        self.lr_a = 3e-4
        self.lr_c = 5e-4

        self.batch_size = 32768  # 建议初始值64-256
        self.buffer = []  # 经验缓冲区
        #self.grad_accum_steps = 4  # 梯度累积步数

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
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s).to(device)
        a = self.actor(s)
        dist = Categorical(a)
        action = dist.sample()             #可采取的action
        log_prob = dist.log_prob(action)   #每种action的概率

        return action.detach().cpu().numpy(), log_prob

    # def learn(self, log_prob, s, s_, rew):
    #     # 确保输入数据在GPU上
    #     if isinstance(s, np.ndarray):
    #         s = torch.FloatTensor(s).to(device)
    #     if isinstance(s_, np.ndarray):
    #         s_ = torch.FloatTensor(s_).to(device)
    #     if isinstance(rew, (float, int)):
    #         rew = torch.FloatTensor([rew]).to(device)

            
    #     #使用Critic网络估计状态值
    #     v = self.critic(s)
    #     v_ = self.critic(s_)

    #     critic_loss = self.loss(self.gamma * v_ + rew, v)
    #     self.critic_optim.zero_grad()
    #     critic_loss.backward()
    #     self.critic_optim.step()

    #     td = self.gamma * v_ + rew - v          #计算TD误差
    #     loss_actor = -log_prob * td.detach()
    #     self.actor_optim.zero_grad()
    #     loss_actor.backward()
    #     self.actor_optim.step()

    def learn(self, samples=None):
        """支持单样本和批量更新"""
        if samples is None:  # 触发批量更新
            if len(self.buffer) < self.batch_size:
                return
            samples = self.buffer
            self.buffer = []
        else:
            samples = [samples]  # 包装单样本
        
        # 批量处理数据
        log_probs = torch.stack([s[0] for s in samples])
        states = torch.stack([torch.FloatTensor(s[1]).to(device) for s in samples])
        next_states = torch.stack([torch.FloatTensor(s[2]).to(device) for s in samples])
        rewards = torch.FloatTensor([s[3] for s in samples]).to(device)
        
        # 批量计算critic损失
        values = self.critic(states)
        next_values = self.critic(next_states)
        critic_loss = self.loss(self.gamma * next_values + rewards, values)
        
        # 批量计算actor损失
        td_errors = (self.gamma * next_values + rewards - values).detach()
        actor_loss = -(log_probs * td_errors).mean()
        
        # 参数更新
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
