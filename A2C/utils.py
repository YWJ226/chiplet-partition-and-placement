import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import random
import sp

class FloorplanOptimizer:
    def __init__(self, areas: List[float], alpha: float = 0.1,
                 beta_range: Tuple[float, float] = (0.5, 2),
                 init_sp1: List[int] = None,
                 init_sp2: List[int] = None,
                 init_widths: List[float] = None,
                 init_heights: List[float] = None):

        self.areas = areas
        self.num_blocks = len(areas)
        self.alpha = alpha
        self.beta_min, self.beta_max = beta_range
        self.w_gap = 0.1
        
        self.ini_sp1 = init_sp1.copy() if init_sp1 is not None else list(range(self.num_blocks))
        self.ini_sp2 = init_sp2.copy() if init_sp2 is not None else list(range(self.num_blocks))
        self.sp1 = self.ini_sp1.copy()  # 再次copy确保独立
        self.sp2 = self.ini_sp2.copy()

        
        if init_widths is not None and init_heights is not None:
            self.ini_widths = np.array(init_widths)
            self.ini_heights = np.array(init_heights)
            self.widths = self.ini_widths.copy()
            self.heights = self.ini_heights.copy()
        else:
            self.betas = np.random.uniform(self.beta_min, self.beta_max, self.num_blocks)
            self.widths = np.sqrt(self.areas / self.betas)
            self.heights = self.areas / self.widths
    
    # 如果未传入初始序列，则随机打乱
        if init_sp1 is None:
            random.shuffle(self.sp1)
        if init_sp2 is None:
            random.shuffle(self.sp2)


        self.positions = np.zeros((self.num_blocks, 2))
        self._update_positions()

        
    def evaluate(self) -> float:
        W, H = self._calculate_contour()
        A = W * H
        L = self._calculate_hpwl()
        return A + self.alpha * L
    
    def _calculate_contour(self) -> Tuple[float, float]:

        left_edges = self.positions[:,0] - self.widths/2
        bottom_edges = self.positions[:,1] - self.heights/2
        right_edges = self.positions[:,0] + self.widths/2
        top_edges = self.positions[:,1] + self.heights/2

        W = max(right_edges) - min(left_edges)
        H = max(top_edges) - min(bottom_edges)
        return W, H
    
    def _calculate_hpwl(self) -> float:
        total_hpwl = 0.0
        # 假设模块间的连接关系是全连接的（实际应用中应根据具体netlist修改）
        for i in range(self.num_blocks):
            for j in range(i+1, self.num_blocks):
                # 计算两个模块间的HPWL
                hpwl = (abs(self.positions[i][0] - self.positions[j][0]) + 
                        abs(self.positions[i][1] - self.positions[j][1]))
                total_hpwl += hpwl
        return total_hpwl
    
    def _check_overlap(self) -> bool:
        for i in range(self.num_blocks):
            for j in range(i+1, self.num_blocks):
                xi, yi = self.positions[i]
                xj, yj = self.positions[j]
                wi, hi = self.widths[i], self.heights[i]
                wj, hj = self.widths[j], self.heights[j]
                
                if not (abs(xi - xj) > (wi + wj)/2 + self.w_gap or 
                        abs(yi - yj) > (hi + hj)/2 + self.w_gap):
                    return True
        return False
    
    def move_operation(self, op_type: int):

        self._save_state()

        if op_type == 0:
            i, j = random.sample(range(self.num_blocks), 2)
            self.sp1[i], self.sp1[j] = self.sp1[j], self.sp1[i]
        elif op_type == 1:
            i, j = random.sample(range(self.num_blocks), 2)
            self.sp2[i], self.sp2[j] = self.sp2[j], self.sp2[i]
        elif op_type == 2:
            # 随机选择两个不同的模块ID
            module_a, module_b = random.sample(range(self.num_blocks), 2)
            # 找到这两个模块在sp1和sp2中的位置索引
            idx_a_sp1 = self.sp1.index(module_a)
            idx_b_sp1 = self.sp1.index(module_b)
            idx_a_sp2 = self.sp2.index(module_a)
            idx_b_sp2 = self.sp2.index(module_b)
            # 交换这两个模块在序列中的位置
            self.sp1[idx_a_sp1], self.sp1[idx_b_sp1] = module_b, module_a
            self.sp2[idx_a_sp2], self.sp2[idx_b_sp2] = module_b, module_a
        elif op_type == 3:
            block = random.randint(0, self.num_blocks-1)
            new_pos = random.randint(0, self.num_blocks-1)
            # 从原位置移除模块
            self.sp1.remove(block)
            self.sp2.remove(block)
            # 在新位置插入模块
            self.sp1.insert(new_pos, block)
            self.sp2.insert(new_pos, block)
        elif op_type == 4:
            block = random.randint(0, self.num_blocks-1)
            self.widths[block], self.heights[block] = self.heights[block], self.widths[block]
        elif op_type == 5:
            block = random.randint(0, self.num_blocks-1)
            new_beta = np.random.uniform(self.beta_min, self.beta_max)
            self.widths[block] = np.sqrt(self.areas[block] / new_beta)
            self.heights[block] = self.areas[block] / self.widths[block]
        elif op_type == 6:
            i, j = random.sample(range(self.num_blocks), 2)
            self.sp1[i], self.sp1[j] = self.sp1[j], self.sp1[i]
            self.sp2[i], self.sp2[j] = self.sp2[j], self.sp2[i] 
        self._update_positions()
        return self.check_aspect_ratio()
    
    
    def _update_positions(self):

        self.spresult, self.sizes = sp.convert_sp_to_layout(self.sp1, self.sp2, self.widths, self.heights, self.w_gap)
        sorted_result = {
        k: self.spresult[k] 
        for k in sorted(self.spresult.keys(), key=lambda x: int(x))
        }
        for module in range(self.num_blocks):
            self.positions[module] = sorted_result[str(module)]
    
    def visualize_2(self):
        fig, ax = plt.subplots()
        for m in self.spresult:
            x, y = self.spresult[m]
            w, h = self.sizes[m]

            rect = plt.Rectangle((x, y), w, h, 
                                fill=False, edgecolor='b', linewidth=1)
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2, m, ha='center', va='center')
        
        ax.set_xlim(0, max(x + w for m, (x, y) in self.spresult.items() for w, h in [self.sizes[m]]) + 1)
        ax.set_ylim(0, max(y + h for m, (x, y) in self.spresult.items() for w, h in [self.sizes[m]]) + 1)
        ax.set_aspect('equal')
        plt.title('Floorplan Visualization')
        plt.show()

    def revert_move(self):
        # 恢复上一次移动前的状态
        self.sp1 = self._prev_sp1.copy()
        self.sp2 = self._prev_sp2.copy()
        self.positions = self._prev_positions.copy()
        self.widths = self._prev_widths.copy()
        self.heights = self._prev_heights.copy()

    def check_aspect_ratio(self):
        """检查当前布局宽高比是否在0.5-0.2之间"""
        W, H = self._calculate_contour() 
        if 2 < W/H :
            return 1
        elif W/H <0.5 :
            return 2
        else :
            return 0
    
    def _save_state(self):
        """保存当前状态用于回滚"""
        self._prev_sp1 = self.sp1.copy()
        self._prev_sp2 = self.sp2.copy()
        self._prev_positions = self.positions.copy()
        self._prev_widths = self.widths.copy()
        self._prev_heights = self.heights.copy()
    
    def reset(self):
        """重置到初始状态，确保不影响原始ini_参数"""
        self.sp1 = self.ini_sp1.copy()  # 使用副本
        self.sp2 = self.ini_sp2.copy()
        self.widths = self.ini_widths.copy()
        self.heights = self.ini_heights.copy()
        self._update_positions()

    def move_top_to_bottomright(self):
        """将最上方模块与随机模块交换位置"""
        self._save_state()
        max_y_idx = np.argmax(self.positions[:,1])
        top_block = self.sp1[max_y_idx]
        
        # 随机选择交换模块(排除自己)
        other_blocks = [b for b in self.sp1 if b != top_block]
        if not other_blocks:  # 只有一个模块的情况
            self._update_positions()
            return self.check_aspect_ratio()
        
        swap_block = random.choice(other_blocks)
        
        # 交换两个模块在sp1和sp2中的位置
        idx1 = self.sp1.index(top_block)
        idx2 = self.sp1.index(swap_block)
        self.sp1[idx1], self.sp1[idx2] = self.sp1[idx2], self.sp1[idx1]
        
        idx1 = self.sp2.index(top_block)
        idx2 = self.sp2.index(swap_block)
        self.sp2[idx1], self.sp2[idx2] = self.sp2[idx2], self.sp2[idx1]
        
        self._update_positions()
        return self.check_aspect_ratio()


    def move_right_to_topleft(self):
        """将最右侧模块与随机模块交换位置"""
        self._save_state()
        max_x_idx = np.argmax(self.positions[:,0])
        right_block = self.sp1[max_x_idx]
        
        # 随机选择交换模块(排除自己)
        other_blocks = [b for b in self.sp1 if b != right_block]
        if not other_blocks:  # 只有一个模块的情况
            self._update_positions()
            return self.check_aspect_ratio()
        
        swap_block = random.choice(other_blocks)
        
        # 交换两个模块在sp1和sp2中的位置
        idx1 = self.sp1.index(right_block)
        idx2 = self.sp1.index(swap_block)
        self.sp1[idx1], self.sp1[idx2] = self.sp1[idx2], self.sp1[idx1]
        
        idx1 = self.sp2.index(right_block)
        idx2 = self.sp2.index(swap_block)
        self.sp2[idx1], self.sp2[idx2] = self.sp2[idx2], self.sp2[idx1]
        
        self._update_positions()
        return self.check_aspect_ratio()





