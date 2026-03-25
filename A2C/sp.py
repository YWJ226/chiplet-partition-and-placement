import collections
import matplotlib.pyplot as plt
import numpy as np

class SequencePairLayout:
    def __init__(self, pos_seq, neg_seq, modules_size):
        """
        pos_seq: list of module names in positive sequence
        neg_seq: list of module names in negative sequence
        modules_size: dict mapping module name to (width, height)
        """
        self.pos_seq = pos_seq
        self.neg_seq = neg_seq
        self.modules_size = modules_size
        self.modules = pos_seq
        self.h_graph = {m: set() for m in self.modules}  # horizontal relations
        self.v_graph = {m: set() for m in self.modules}  # vertical relations
        self.x = {}  # x-coordinates
        self.y = {}  # y-coordinates

    def build_constraint_graphs(self):
        # compute horizontal constraints: for every pair a,b, if a before b in pos AND a before b in neg = no relation;
        # if a before b in pos but b before a in neg -> a is left of b
        # if a before b in neg but b before a in pos -> a is below b
        pos_index = {m: i for i, m in enumerate(self.pos_seq)}
        neg_index = {m: i for i, m in enumerate(self.neg_seq)}
        n = len(self.modules)
        for i in range(n):
            for j in range(i+1, n):
                a = self.modules[i]
                b = self.modules[j]
                # check positions
                if pos_index[a] < pos_index[b] and neg_index[a] < neg_index[b]:
                    # a is to left of b
                    self.h_graph[a].add(b)
                elif pos_index[a] > pos_index[b] and neg_index[a] > neg_index[b]:
                    # b is to left of a => edge b->a
                    self.h_graph[b].add(a)
                elif pos_index[a] < pos_index[b] and neg_index[a] > neg_index[b]:
                    # a above b
                    self.v_graph[b].add(a)
                elif pos_index[a] > pos_index[b] and neg_index[a] < neg_index[b]:
                    # b above a
                    self.v_graph[a].add(b)

    def longest_path(self, graph, size_map, gap=0):
        # compute longest path distances in DAG
        indeg = {u: 0 for u in graph}
        for u in graph:
            for v in graph[u]:
                indeg[v] += 1
        # initialize queue
        dq = collections.deque([u for u, d in indeg.items() if d == 0])
        dist = {u: 0 for u in graph}
        order = []
        while dq:
            u = dq.popleft()
            order.append(u)
            for v in graph[u]:
                # update distance: position of v = max(current, position of u + width(u) + gap)
                dist[v] = max(dist[v], dist[u] + size_map[u] + gap)
                indeg[v] -= 1
                if indeg[v] == 0:
                    dq.append(v)
        return dist

    def layout(self, w_gap=0, h_gap=0):
        self.build_constraint_graphs()
        # horizontal: width
        width_map = {m: self.modules_size[m][0] for m in self.modules}
        x_coords = self.longest_path(self.h_graph, width_map, w_gap)
        # vertical: height
        height_map = {m: self.modules_size[m][1] for m in self.modules}
        y_coords = self.longest_path(self.v_graph, height_map, h_gap)
        # save results
        self.x = x_coords
        self.y = y_coords
        return {m: (self.x[m], self.y[m]) for m in self.modules}


def convert_sp_to_layout(sp1, sp2, widths, heights, gap = 0):

    # 转换为字符串列表
    pos = [str(x) for x in sp1]
    neg = [str(x) for x in sp2]
    
    # 构建尺寸字典
    sizes = {
        str(i): (float(w), float(h)) 
        for i, (w, h) in enumerate(zip(widths, heights))
    }
    w_gap = gap
    h_gap = gap
    sp = SequencePairLayout(pos, neg, sizes)
    result = sp.layout(w_gap, h_gap) 

    # fig, ax = plt.subplots()
    # for m in result:
    #     x, y = result[m]
    #     w, h = sizes[m]

    #     rect = plt.Rectangle((x, y), w, h, 
    #                         fill=False, edgecolor='b', linewidth=1)
    #     ax.add_patch(rect)
    #     ax.text(x + w/2, y + h/2, m, ha='center', va='center')
    
    # ax.set_xlim(0, max(x + w for m, (x, y) in result.items() for w, h in [sizes[m]]) + 1)
    # ax.set_ylim(0, max(y + h for m, (x, y) in result.items() for w, h in [sizes[m]]) + 1)
    # ax.set_aspect('equal')
    # plt.title('Floorplan Visualization')
    # plt.show()

    return result, sizes


# Example usage
if __name__ == '__main__':
    # define sequences and module sizes
    pos = ['D', 'A', 'C', 'F', 'E', 'B']
    neg = ['F', 'A', 'B', 'C', 'D', 'E']
    sizes = {'A': (0.5, 0.3), 'B': (0.6, 0.2), 'C': (0.4, 0.3), 'D': (0.9, 0.3), 'E':(0.4, 0.7), 'F':(0.7, 0.3)}
    sp = SequencePairLayout(pos, neg, sizes)
    result = sp.layout(w_gap=0.1, h_gap=0.05)  # 设置水平间距0.1，垂直间距0.05



