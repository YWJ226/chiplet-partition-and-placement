import gym
from gym import spaces
from model import Actor_Critic
from utils import FloorplanOptimizer
import numpy as np
import torch
import random
from typing import Optional, List, Dict, Any
import evaluation
import json
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import tempfile
import shutil
import imageio.v2 as imageio
# 检测GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VisualizationRecorder:
    """记录每次扰动并生成可视化视频的类"""
    
    def __init__(self, record_all_steps=False, save_interval=10):
        """
        初始化记录器
        
        Args:
            record_all_steps: 是否记录所有步骤，False则只记录关键步骤
            save_interval: 保存间隔，每N步保存一次
        """
        self.record_all_steps = record_all_steps
        self.save_interval = save_interval
        self.records = []
        self.current_episode = 0
        self.current_step = 0
        self.best_area = float('inf')
        self.best_hpwl = float('inf')
        self.best_episode_records = []  # 记录最优episode的扰动过程
        self.best_episode = -1
        self.current_episode_records = []  # 当前episode的记录
        
    def record_step(self, optimizer, action, accepted, episode, step):
        """
        记录一次扰动步骤
        
        Args:
            optimizer: FloorplanOptimizer实例
            action: 执行的动作
            accepted: 是否接受扰动
            episode: 当前episode
            step: 当前step
        """
        self.current_episode = episode
        self.current_step = step
        
        # 计算当前面积和HPWL
        w, h = optimizer._calculate_contour()
        area = w * h
        hpwl = optimizer._calculate_hpwl()
        
        # 更新最优值
        if area < self.best_area:
            self.best_area = area
            self.best_episode = episode
        if hpwl < self.best_hpwl:
            self.best_hpwl = hpwl
        
        # 决定是否记录这一步
        should_record = (
            self.record_all_steps or
            accepted or  # 接受扰动时记录
            step % self.save_interval == 0 or  # 定期记录
            area <= self.best_area * 1.1  # 接近最优时记录
        )
        
        if should_record:
            record = {
                "episode": episode,
                "step": step,
                "positions": optimizer.positions.tolist(),
                "widths": optimizer.widths.tolist(),
                "heights": optimizer.heights.tolist(),
                "area": area,
                "hpwl": hpwl,
                "accepted": accepted,
                "action": action,
                "best_area": self.best_area,
                "best_hpwl": self.best_hpwl,
                "spresult": {str(k): list(v) for k, v in optimizer.spresult.items()},
                "sizes": {str(k): list(v) for k, v in optimizer.sizes.items()}
            }
            self.records.append(record)
            
            # 如果是当前最优episode，记录到最优episode记录中
            if episode == self.best_episode:
                self.best_episode_records.append(record)
    
    def save_data(self, filename):
        """保存记录数据到JSON文件"""
        os.makedirs('visualization_data', exist_ok=True)
        filepath = f"visualization_data/{filename}"
        
        # 确保所有数据都可以被JSON序列化
        serializable_records = []
        for record in self.records:
            serializable_record = {}
            for key, value in record.items():
                # 处理numpy数据类型
                if isinstance(value, (np.integer, np.floating)):
                    serializable_record[key] = float(value)
                elif isinstance(value, np.ndarray):
                    serializable_record[key] = value.tolist()
                elif isinstance(value, (bool, int, float, str, list, dict)) or value is None:
                    serializable_record[key] = value
                else:
                    # 转换为字符串作为备用方案
                    serializable_record[key] = str(value)
            serializable_records.append(serializable_record)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_records, f, indent=2, ensure_ascii=False)
        print(f"可视化数据已保存到 {filepath}")
    
    def generate_video(self, filename, dpi=100, target_duration=60, frame_duration=1000, video_format="mp4"):
        """
        生成可视化视频
        
        Args:
            filename: 输出视频文件名
            dpi: 图像分辨率
            target_duration: 目标视频时长（秒）
            frame_duration: 每帧显示时长（毫秒）
            video_format: 视频格式，支持 "mp4" 或 "gif"
        """
        # 使用最优episode的记录生成视频
        if not self.best_episode_records:
            print("没有最优episode的记录数据，无法生成视频")
            return
        
        # print(f"开始生成最优episode (Episode {self.best_episode}) 的可视化视频...")
        
        # 根据格式选择生成方法
        if video_format.lower() == "mp4":
            self._create_mp4_animation(filename, dpi, target_duration)
        else:
            self._create_best_episode_animation(filename, dpi, target_duration, frame_duration)
        
        print(f"最优episode视频已生成: {filename}")
    
    def _create_frame(self, record, output_path, dpi):
        """创建单个帧图像"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 布局图 - 使用spresult和sizes字典确保正确布局
        spresult = record.get("spresult", {})
        sizes = record.get("sizes", {})
        
        if spresult and sizes:
            # 使用spresult和sizes字典绘制布局
            for m in spresult:
                x, y = spresult[m]
                w, h = sizes[m]
                
                rect = Rectangle((x, y), w, h, 
                               fill=False, edgecolor='blue', linewidth=1)
                ax.add_patch(rect)
                ax.text(x + w/2, y + h/2, str(m), 
                        ha='center', va='center', fontsize=8)
            
            # 设置布局图范围
            margin = 1.0
            x_min = min(x for x, y in spresult.values()) - margin
            x_max = max(x + w for m, (x, y) in spresult.items() for w, h in [sizes[m]]) + margin
            y_min = min(y for x, y in spresult.values()) - margin
            y_max = max(y + h for m, (x, y) in spresult.items() for w, h in [sizes[m]]) + margin
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        else:
            # 回退到原来的方法（兼容性）
            positions = np.array(record["positions"])
            widths = np.array(record["widths"])
            heights = np.array(record["heights"])
            
            for i in range(len(positions)):
                x = positions[i][0] - widths[i] / 2
                y = positions[i][1] - heights[i] / 2
                rect = Rectangle((x, y), widths[i], heights[i], 
                               fill=False, edgecolor='blue', linewidth=1)
                ax.add_patch(rect)
                ax.text(x + widths[i]/2, y + heights[i]/2, str(i), 
                        ha='center', va='center', fontsize=8)
            
            # 设置布局图范围
            margin = 0.5
            x_min = min(positions[:, 0] - widths/2) - margin
            x_max = max(positions[:, 0] + widths/2) + margin
            y_min = min(positions[:, 1] - heights/2) - margin
            y_max = max(positions[:, 1] + heights/2) + margin
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
        ax.set_aspect('equal')
        ax.set_title(f'Floorplan (Episode {record["episode"]}, Step {record["step"]})')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # 添加信息文本
        info_text = (f"Current Area: {record['area']:.2f}\n"
                    f"Current HPWL: {record['hpwl']:.2f}\n"
                    f"Best Area: {record['best_area']:.2f}\n"
                    f"Best HPWL: {record['best_hpwl']:.2f}\n"
                    f"Action: {record['action']}\n"
                    f"Accepted: {'Yes' if record['accepted'] else 'No'}")
        
        plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def _create_best_episode_animation(self, output_path, dpi=100, target_duration=60, frame_duration=1000):
        """生成最优episode的动画"""
        # 使用最优episode的记录
        original_records = self.best_episode_records
        
        # 计算目标帧数：60秒视频，每帧1秒，总共60帧
        target_frames = target_duration  # 因为每帧1秒，所以帧数等于秒数
        
        # 从最优episode记录中均匀采样60帧
        if len(original_records) > target_frames:
            # 计算采样步长
            step = max(1, len(original_records) // target_frames)
            sampled_records = original_records[::step]
            
            # 确保正好60帧
            if len(sampled_records) > target_frames:
                sampled_records = sampled_records[:target_frames]
            elif len(sampled_records) < target_frames:
                # 如果采样后帧数不足，从原始记录中均匀补充
                additional_step = len(original_records) // (target_frames - len(sampled_records))
                additional_indices = list(range(0, len(original_records), additional_step))[:target_frames - len(sampled_records)]
                additional_records = [original_records[i] for i in additional_indices]
                sampled_records.extend(additional_records)
            
            print(f"从 {len(original_records)} 个扰动结果中采样 {len(sampled_records)} 帧，每帧显示1秒，总时长: {target_duration}秒")
        else:
            sampled_records = original_records
            print(f"使用全部 {len(sampled_records)} 个扰动结果，每帧显示1秒，总时长: {len(sampled_records)}秒")
        
        # print(f"生成最优episode (Episode {self.best_episode}) 的 {len(sampled_records)} 帧GIF动画...")
        
        # 创建临时目录存储帧图像
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 生成所有帧
            frames = []
            for i, record in enumerate(sampled_records):
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                self._create_frame(record, frame_path, dpi)
                frames.append(frame_path)
            
            # 使用PIL库创建GIF，确保不循环播放
            try:
                from PIL import Image
                
                # 打开所有帧图像
                images = [Image.open(frame) for frame in frames]
                
                # 保存为GIF，设置每帧显示1秒，不循环
                images[0].save(
                    output_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=frame_duration,  # 每帧显示1000毫秒（1秒）
                    loop=0  # 0表示不循环
                )
                
            except ImportError:
                # 如果PIL不可用，回退到imageio
                print("PIL库不可用，使用imageio创建GIF")
                with imageio.get_writer(output_path, mode='I', fps=1) as writer:  # 1fps = 每帧1秒
                    for frame_path in frames:
                        image = imageio.imread(frame_path)
                        writer.append_data(image)
                    
        finally:
            # 清理临时文件
            shutil.rmtree(temp_dir)
    
    def _create_mp4_animation(self, output_path, dpi=100, target_duration=60):
        """生成MP4格式的动画"""
        # 使用最优episode的记录
        original_records = self.best_episode_records
        
        # 计算目标帧数：60秒视频，3fps帧率，总共180帧
        target_frames = target_duration * 3  # 3fps × 60秒 = 180帧
        
        # 从最优episode记录中均匀采样180帧
        if len(original_records) > target_frames:
            # 计算采样步长
            step = max(1, len(original_records) // target_frames)
            sampled_records = original_records[::step]
            
            # 确保正好180帧
            if len(sampled_records) > target_frames:
                sampled_records = sampled_records[:target_frames]
            elif len(sampled_records) < target_frames:
                # 如果采样后帧数不足，从原始记录中均匀补充
                additional_step = len(original_records) // (target_frames - len(sampled_records))
                additional_indices = list(range(0, len(original_records), additional_step))[:target_frames - len(sampled_records)]
                additional_records = [original_records[i] for i in additional_indices]
                sampled_records.extend(additional_records)
            
            print(f"从 {len(original_records)} 个扰动结果中采样 {len(sampled_records)} 帧，3fps帧率，总时长: {target_duration}秒")
        else:
            sampled_records = original_records
            print(f"使用全部 {len(sampled_records)} 个扰动结果，3fps帧率，总时长: {len(sampled_records)/3:.1f}秒")
        
        # print(f"生成最优episode (Episode {self.best_episode}) 的 {len(sampled_records)} 帧MP4动画...")
        
        # 创建临时目录存储帧图像
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 生成所有帧（每帧显示单个扰动结果）
            frames = []
            for i, record in enumerate(sampled_records):
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                self._create_frame(record, frame_path, dpi)
                frames.append(frame_path)
            
            # 确保所有图像尺寸和通道数一致
            from PIL import Image
            images = [Image.open(frame) for frame in frames]
            
            # 获取最大尺寸
            max_width = max(img.width for img in images)
            max_height = max(img.height for img in images)
            
            # 确保尺寸能被16整除以避免警告
            max_width = (max_width + 15) // 16 * 16
            max_height = (max_height + 15) // 16 * 16
            
            # 调整所有图像到相同尺寸和通道数
            resized_frames = []
            for i, img in enumerate(images):
                # 确保图像是RGB模式
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 调整尺寸
                if img.width != max_width or img.height != max_height:
                    # 创建新图像，保持宽高比，填充到最大尺寸
                    new_img = Image.new('RGB', (max_width, max_height), 'white')
                    new_img.paste(img, ((max_width - img.width) // 2, (max_height - img.height) // 2))
                    resized_path = os.path.join(temp_dir, f"resized_frame_{i:06d}.png")
                    new_img.save(resized_path)
                    resized_frames.append(resized_path)
                else:
                    resized_frames.append(frames[i])
            
            # 使用imageio创建MP4，3fps = 每帧显示0.33秒
            with imageio.get_writer(output_path, mode='I', fps=3) as writer:
                for frame_path in resized_frames:
                    image = imageio.imread(frame_path)
                    writer.append_data(image)
                    
        finally:
            # 清理临时文件
            shutil.rmtree(temp_dir)
    
    def _create_double_frame(self, record1, record2, output_path, dpi):
        """创建包含两张图片的帧"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 第一张图片
        self._draw_layout(ax1, record1)
        ax1.set_title(f'Step {record1["step"]} - Before')
        
        # 第二张图片
        self._draw_layout(ax2, record2)
        ax2.set_title(f'Step {record2["step"]} - After')
        
        # 添加总体信息
        info_text = (f"Episode: {record1['episode']}\n"
                    f"Action: {record1['action']}\n"
                    f"Accepted: {'Yes' if record1['accepted'] else 'No'}\n"
                    f"Area: {record1['area']:.2f} → {record2['area']:.2f}\n"
                    f"HPWL: {record1['hpwl']:.2f} → {record2['hpwl']:.2f}")
        
        plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def _draw_layout(self, ax, record):
        """在指定轴上绘制布局"""
        # 布局图 - 使用spresult和sizes字典确保正确布局
        spresult = record.get("spresult", {})
        sizes = record.get("sizes", {})
        
        if spresult and sizes:
            # 使用spresult和sizes字典绘制布局
            for m in spresult:
                x, y = spresult[m]
                w, h = sizes[m]
                
                rect = Rectangle((x, y), w, h, 
                               fill=False, edgecolor='blue', linewidth=1)
                ax.add_patch(rect)
                ax.text(x + w/2, y + h/2, str(m), 
                        ha='center', va='center', fontsize=8)
            
            # 设置布局图范围
            margin = 1.0
            x_min = min(x for x, y in spresult.values()) - margin
            x_max = max(x + w for m, (x, y) in spresult.items() for w, h in [sizes[m]]) + margin
            y_min = min(y for x, y in spresult.values()) - margin
            y_max = max(y + h for m, (x, y) in spresult.items() for w, h in [sizes[m]]) + margin
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        else:
            # 回退到原来的方法（兼容性）
            positions = np.array(record["positions"])
            widths = np.array(record["widths"])
            heights = np.array(record["heights"])
            
            for i in range(len(positions)):
                x = positions[i][0] - widths[i] / 2
                y = positions[i][1] - heights[i] / 2
                rect = Rectangle((x, y), widths[i], heights[i], 
                               fill=False, edgecolor='blue', linewidth=1)
                ax.add_patch(rect)
                ax.text(x + widths[i]/2, y + heights[i]/2, str(i), 
                        ha='center', va='center', fontsize=8)
            
            # 设置布局图范围
            margin = 0.5
            x_min = min(positions[:, 0] - widths/2) - margin
            x_max = max(positions[:, 0] + widths/2) + margin
            y_min = min(positions[:, 1] - heights/2) - margin
            y_max = max(positions[:, 1] + heights/2) + margin
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
        ax.set_aspect('equal')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
    
    def _create_optimized_animation(self, output_path, fps=10, dpi=100):
        """优化的动画生成方法，生成GIF动画"""
        # 如果记录太多，进行采样以减少动画长度
        max_frames = 30  # 最大帧数
        if len(self.records) > max_frames:
            step = len(self.records) // max_frames
            sampled_records = self.records[::step]
            if len(sampled_records) < max_frames:
                sampled_records = self.records[:max_frames]
        else:
            sampled_records = self.records
        
        print(f"生成 {len(sampled_records)} 帧GIF动画...")
        
        # 创建临时目录存储帧图像
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 生成所有帧
            frames = []
            for i, record in enumerate(sampled_records):
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                self._create_frame(record, frame_path, dpi)
                frames.append(frame_path)
            
            # 使用imageio创建GIF，设置loop=0确保不循环播放
            with imageio.get_writer(output_path, mode='I', fps=fps, loop=0) as writer:
                for frame_path in frames:
                    image = imageio.imread(frame_path)
                    writer.append_data(image)
                    
        finally:
            # 清理临时文件
            shutil.rmtree(temp_dir)
    
    def clear_records(self):
        """清空记录"""
        self.records = []
        self.best_area = float('inf')
        self.best_hpwl = float('inf')

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
                init_heights: Optional[List[float]] = None,
                recorder: Optional[VisualizationRecorder] = None):
        
        self.init_areas = areas
        self.init_sp1 = init_sp1
        self.init_sp2 = init_sp2 
        self.init_widths = init_widths
        self.init_heights = init_heights
        self.recorder = recorder
        
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
        self.current_episode = 0
        self.current_step = 0
        
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

        # 记录扰动步骤
        if self.recorder:
            self.recorder.record_step(self.optimizer, action, accept, self.current_episode, self.current_step)
            self.current_step += 1

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


def train(chip_name="ga102", record_visualization=False, record_all_steps=False, save_interval=10):
    # 设置所有随机种子
    seed = 41  # 可以设置为任意整数
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 创建可视化记录器
    recorder = None
    if record_visualization:
        recorder = VisualizationRecorder(record_all_steps=record_all_steps, save_interval=save_interval)
        print(f"启用可视化记录 - 记录所有步骤: {record_all_steps}, 保存间隔: {save_interval}")

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
        init_heights=init_heights,
        recorder=recorder
    )
    agent = Actor_Critic(env)  # GPU支持已在类内部处理
    
    # 训练参数
    episodes = 20 #400
    print_interval = 10
    
    minarea = 1000
    minwl = 10000
    # 训练循环
    for ep in range(episodes):
        state = env.reset()
        env.current_episode = ep  # 设置当前episode
        env.current_step = 0      # 重置step计数
        
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            steps += 1
            if steps > 840:  # 安全限制
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

        # if ep % print_interval == 0:
        #     print(f"Episode {ep}, Total Reward: {total_reward:.2f}")
        #     print(f" Area: {w*h}, Hpwl: {l}")
        #     print(f" W,H: {w, h}")
        # if ep == episodes-1  :
        #     print(f" Area: {w*h}, Hpwl: {l}")
        #     env.optimizer.visualize_2()
    
    # 保存结果和可视化数据
    power, perf = evaluation.eva_power(conn_name, ds_name, partition, env.optimizer)
    save_floorplan_results(chip_name, partition, env.optimizer, power, perf, minarea, minwl)
    
    # 保存和生成可视化数据
    if recorder and recorder.records:
        # 保存记录数据
        data_filename = f"{chip_name}_visualization_data.json"
        recorder.save_data(data_filename)
        
        # 生成MP4视频，使用3fps帧率，每帧显示单个扰动结果
        video_filename = f"{chip_name}_floorplan_animation.mp4"
        recorder.generate_video(video_filename, dpi=100, target_duration=60, frame_duration=333, video_format="mp4")
        
        print(f"可视化数据已保存: {data_filename}")
        print(f"可视化视频已生成: {video_filename}")
    
    print (f"miniarea :{minarea}, minwl:{minwl}")
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
    parser = argparse.ArgumentParser(description='布图优化强化学习训练')
    parser.add_argument('--chip', type=str, default='ga102', 
                       help='要处理的芯片名称,默认为ga102')
    parser.add_argument('--all', action='store_true',
                      help='处理partition_output.json中的所有芯片')
    parser.add_argument('--record-visualization', action='store_true',
                      help='启用扰动记录和视频生成')
    parser.add_argument('--record-all-steps', action='store_true',
                      help='记录所有步骤(默认只记录关键步骤)')
    parser.add_argument('--save-interval', type=int, default=10,
                      help='记录保存间隔(默认每10步保存一次)')
    parser.add_argument('--video-fps', type=int, default=10,
                      help='视频帧率(默认10fps)')
    parser.add_argument('--video-dpi', type=int, default=100,
                      help='视频分辨率(默认100dpi)')

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
            train(
                chip_name=chip,
                record_visualization=args.record_visualization,
                record_all_steps=args.record_all_steps,
                save_interval=args.save_interval
            )
        except Exception as e:
            print(f"处理芯片 {chip} 时出错: {e}")
            continue
