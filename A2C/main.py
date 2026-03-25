import gym
from gym import spaces
from model import Actor_Critic
from utils import FloorplanOptimizer
import numpy as np
import torch
import random
from typing import Optional, List 
import evaluation
import json
import argparse
import os

# 检测GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_chip_data(chip_name="ga102"):
    """从partition_output.json加载芯片数据"""
    try:
        with open('../gap-main/partition_output.json', 'r') as f:
            data = json.load(f)
        
        chip_data = data['results'].get(chip_name)
        if not chip_data:
            raise ValueError(f"芯片 {chip_name} 不存在于输出文件中")
            
        return {
            'areas': list(chip_data['partition_areas'].values()),
            'partition': chip_data['partition_assignment'],
            'conn_name': chip_name,
            'ds_name': chip_data['family']
        }
    except Exception as e:
        print(f"加载芯片数据失败: {e}")
        return None


class FloorplanEnv:
    """将布图优化问题封装为强化学习环境"""
    def __init__(self, 
                areas: List[float],
                init_sp1: Optional[List[int]] = None,
                init_sp2: Optional[List[int]] = None,
                init_widths: Optional[List[float]] = None,
                init_heights: Optional[List[float]] = None):
        
        self.init_areas = areas
        self.init_sp1 = init_sp1
        self.init_sp2 = init_sp2 
        self.init_widths = init_widths
        self.init_heights = init_heights
        
        self.optimizer = FloorplanOptimizer(
            areas=areas,
            init_sp1=init_sp1,
            init_sp2=init_sp2,
            init_widths=init_widths,
            init_heights=init_heights
        )

        self.action_space = spaces.Discrete(9)  # 9种离散动作
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(len(areas)*3,),  # 每个模块的面积、位置x、位置y
            dtype=np.float32
        )
        self.state_dim = len(areas) * 3  # 保持向后兼容
        
    def reset(self):
        """重置环境到初始状态"""
        self.optimizer.reset()
        return self._get_state()

    
    def step(self, action):
        # 保存当前状态和评估值
        # old_state = self._get_state()
        old_value = self.optimizer.evaluate()
        
        # 执行移动操作
        # move_success = True
        # com = 0
        # while   move_success and  com < 50:
        if action == 7:
            move_success = self.optimizer.move_top_to_bottomright()
        elif action == 8:
            move_success = self.optimizer.move_right_to_topleft()
        else:
            move_success = self.optimizer.move_operation(action)

            
            # com += 1
            # if com > 40:
            #     print ("com > 40\n\n")



        new_value = self.optimizer.evaluate()
        accept = new_value < old_value

        if not accept:
            # 拒绝则恢复原状态
            self.optimizer.revert_move()
            reward = 0  # 使用原评估值作为奖励
            if  move_success:
                reward = -0.1
        else:
            reward = (old_value - new_value) / 1500  # 使用新评估值作为奖励
            
        # 添加终止条件：当布局评估值足够小或达到最大步数时终止
        done = new_value < 700  # 750是目标阈值，可根据实际情况调整

        return self._get_state(), reward, done


        
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


def train(chip_name="ga102"):
    # 设置所有随机种子
    seed = 41  # 可以设置为任意整数
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    chip_data = load_chip_data(chip_name)
    if not chip_data:
        print("使用默认芯片数据")
        chip_data = {
            'areas': [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 37, 44, 7, 50],
            'partition': [11, 11, 11, 11, 11, 11, 11, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9, 9, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 2, 12, 12, 12, 12, 12, 12, 12, 3, 3, 3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 13, 13, 13, 13, 13, 13, 15, 13, 10, 10, 10, 10, 10, 14, 8, 10, 10, 14, 14, 14, 8, 8, 8, 8, 8, 8],
            'conn_name': "ga102",
            'ds_name': "Nvidia"
        }

    areas = chip_data['areas']
    partition = chip_data['partition']
    conn_name = chip_data['conn_name']
    ds_name = chip_data['ds_name']



    # 示例芯粒面积
    # areas = [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 37, 44, 7, 50]
    # partition = [11, 11, 11, 11, 11, 11, 11, 0, 
    #            0, 0, 0, 0, 0, 0, 9, 9, 
    #            9, 9, 9, 9, 9, 4, 4, 4, 
    #            4, 4, 4, 4, 1, 1, 1, 1, 
    #            1, 1, 1, 6, 6, 6, 6, 6, 
    #            6, 6, 2, 2, 2, 2, 2, 2, 
    #            2, 12, 12, 12, 12, 12, 12, 12, 
    #            3, 3, 3, 3, 3, 3, 3, 7, 
    #            7, 7, 7, 7, 7, 7, 5, 5, 
    #            5, 5, 5, 5, 5, 13, 13, 13, 
    #            13, 13, 13, 15, 13, 10, 10, 10, 
    #            10, 10, 14, 8, 10, 10, 14, 14, 
    #            14, 8, 8, 8, 8, 8, 8]
    # conn_name = "ga102"  # 连接关系名称
    # ds_name="Nvidia"
    num_blocks = len(areas)
    init_sp1 = list(range(num_blocks))  # [0,1,2,...]
    init_sp2 = list(range(num_blocks))  # [0,1,2,...]
    random.shuffle(init_sp1)
    random.shuffle(init_sp2)
    
    # 2. 计算初始长宽 (使用固定beta值1.0)
    beta = 1.0  # 长宽比为1:1
    init_widths = np.sqrt(np.array(areas) / beta)
    init_heights = np.array(areas) / init_widths

    # 创建环境和Agent
    env = FloorplanEnv(
        areas=areas,
        init_sp1=init_sp1,
        init_sp2=init_sp2,
        init_widths=init_widths,
        init_heights=init_heights
    )
    agent = Actor_Critic(env)  # GPU支持已在类内部处理
    
    # 训练参数
    episodes = 200 #400
    print_interval = 10
    
    minarea = 1000
    minwl = 10000
    # 训练循环
    for ep in range(episodes):
        state = env.reset()
        # env.optimizer.visualize_2()
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            steps += 1
            if steps > 1000:  # 安全限制
                #print(f"达到最大步数限制(1000),终止当前episode")
                break
            action, log_prob = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(log_prob, state, next_state, reward)
            
            state = next_state
            total_reward += reward

        l = env.optimizer._calculate_hpwl()
        w,h = env.optimizer._calculate_contour()
        if w*h < minarea :
            minarea = w*h
            resultep = ep
            if l< minwl:
                minwl = l



        if ep % print_interval == 0:
            print(f"Episode {ep}, Total Reward: {total_reward:.2f}")
            # l = env.optimizer._calculate_hpwl()
            print(f" Area: {w*h}, Hpwl: {l}")
            print(f" W,H: {w, h}")
            # env.optimizer.visualize_2()
        if ep == episodes-1  :
            # l = env.optimizer._calculate_hpwl()
            print(f" Area: {w*h}, Hpwl: {l}")
            env.optimizer.visualize_2()
    power, perf = evaluation.eva_power(conn_name, ds_name, partition, env.optimizer)
    save_floorplan_results(chip_name, partition, env.optimizer, power, perf, minarea, minwl)
    # env.optimizer.visualize_2()
    print (f"miniarea :{minarea}, minwl:{minwl}, ep{resultep}\n")
    print (f"power {power}, perf{perf}")

def save_floorplan_results(chip_name, partition, optimizer, power, perf, minarea, minwl):
    """保存floorplan结果到JSON文件"""
    result = {
        "chip_name": chip_name,
        "partition": partition,
        "positions": optimizer.positions.tolist(),
        "widths": optimizer.widths.tolist(),
        "heights": optimizer.heights.tolist(),
        "power": power,
        "performance": perf,
        # "total_area": optimizer._calculate_contour()[0] * optimizer._calculate_contour()[1],
        # "hpwl": optimizer._calculate_hpwl()
        "total_area": minarea,
        "hpwl": minwl
    }
    
    # 创建results目录如果不存在
    os.makedirs('results', exist_ok=True)
    
    # 保存到文件
    filename = f"results/{chip_name}_floorplan.json"
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"结果已保存到 {filename}")


if __name__ == "__main__":
    # train()

    parser = argparse.ArgumentParser()
    parser.add_argument('--chip', type=str, default='ga102', 
                       help='要处理的芯片名称,默认为ga102')
    parser.add_argument('--all', action='store_true',
                      help='处理partition_output.json中的所有芯片')

    args = parser.parse_args()
    if args.all:
        # 自动处理所有芯片
        with open('../gap-main/partition_output.json', 'r') as f:
            data = json.load(f)
        chip_names = list(data['results'].keys())
    else:
        # 处理指定的芯片列表
        chip_names = args.chip.split(',')
    
    # 为每个芯片运行训练
    for chip in chip_names:
        print(f"\n{'='*40}")
        print(f"开始处理芯片: {chip}")
        print(f"{'='*40}\n")
        
        try:
            train(chip_name=chip)
        except Exception as e:
            print(f"处理芯片 {chip} 时出错: {e}")
            continue
    
    # train(chip_name=args.chip)


