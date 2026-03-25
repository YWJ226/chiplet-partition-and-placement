import math
from typing import List
import networkx as nx
import numpy as np
from dataset.data import get_dataset
from rectpack import newPacker

def get_pp_final(bdg: nx.DiGraph, ptt: List[List[int]], optimizer):
    """
        Return (power, performance)
        pm: partition matrix
    """
    assert sum(map(len, ptt)) == len(bdg)

    num_cpl = len(ptt)
    traffic_total = sum([attr["comm"] for _, __, attr in bdg.edges(data=True)]) * 8
    areas = [0] * num_cpl
    connection_matrix = [[0] * num_cpl for _ in range(num_cpl)]  # number of wire from chiplet i to chiplet j
    traffic_matrix = np.zeros(shape=(num_cpl, num_cpl))  # traffic (Gbit/s) from chiplet i to chiplet j

    for i in range(len(ptt)):
        for j in range(len(ptt[i])):
            areas[i] += bdg.nodes[ptt[i][j]]["block"].area

    for i_a in range(len(ptt)):
        for j_a in range(len(ptt[i_a])):
            for i_b in range(len(ptt)):
                for j_b in range(len(ptt[i_b])):
                    if i_a != i_b and bdg.has_edge(ptt[i_a][j_a], ptt[i_b][j_b]):
                        connection_matrix[i_a][i_b] += math.ceil(bdg[ptt[i_a][j_a]][ptt[i_b][j_b]]["comm"] * 8)
                        traffic_matrix[i_a][i_b] += bdg[ptt[i_a][j_a]][ptt[i_b][j_b]]["comm"] * 8

    tree = convert_positions_to_tree(optimizer)
    global net, s, t, wire_count
    net, s, t, wire_count = get_connections(connection_matrix=connection_matrix)
    wl_current, wl_matrix = compute_wirelength(tree=tree, connection_matrix=connection_matrix)

    print(f"wl_matrix type: {type(wl_matrix)}")
    print(f"wl_matrix shape: {np.array(wl_matrix).shape if hasattr(wl_matrix, '__len__') else 'scalar'}")
    print(f"connection_matrix shape: {np.array(connection_matrix).shape}")
    print(f"traffic_matrix shape: {traffic_matrix.shape}")


    power = 0
    perf = 0

    lmax_cycle = 10  # maximum distance per cycle
    ener_eff_io = 0.59
    ener_eff_wl = lambda wirelength: 2.583171 / 64 * wirelength
    ener_eff_reg = 2.150397 / 128
    for i in range(num_cpl):
        for j in range(num_cpl):
            power += traffic_matrix[i][j] * (ener_eff_io + ener_eff_wl(wl_matrix[i][j]) +
                                             math.floor(wl_matrix[i][j] / lmax_cycle) * ener_eff_reg) / 1000
            perf += (2 + math.floor(wl_matrix[i][j] / lmax_cycle)) * traffic_matrix[i][j] / traffic_total

    return power, perf

def convert_positions_to_tree(optimizer):
    """Convert FloorplanOptimizer positions to tree format for compute_wirelength"""
    class SimpleTree:
        pass
    tree = SimpleTree()
    tree.x_arr = optimizer.positions[:, 0]
    tree.y_arr = optimizer.positions[:, 1] 
    tree.width_arr = optimizer.widths
    tree.height_arr = optimizer.heights
    tree.ind_arr = list(range(len(optimizer.positions)))
    return tree

def compute_wirelength(tree, connection_matrix):
    spacing_ = 0.1
    # length_per_wire value, do not normalize
    total_wirelength = 0
    num_cpl = len(tree.ind_arr)
    wlm = [[0] * num_cpl for _ in range(num_cpl)]
    for i in range(net):
        s_index = tree.ind_arr.index(s[i])
        t_index = tree.ind_arr.index(t[i])
        # wirelength = (abs(tree.x_arr[s_index] + tree.width_arr[s_index] / 2 - tree.x_arr[t_index] - tree.width_arr[t_index] / 2)
        #               + abs(tree.y_arr[s_index] + tree.height_arr[s_index] / 2 - tree.y_arr[t_index] -
        #                     tree.height_arr[t_index] / 2)) * connection_matrix[s_index][t_index]
        x_overlap, y_overlap = False, False
        if tree.x_arr[s_index] >= tree.x_arr[t_index] + tree.width_arr[t_index] or tree.x_arr[
                t_index] >= tree.x_arr[s_index] + tree.width_arr[s_index]:
            dx = abs(tree.x_arr[s_index] + tree.width_arr[s_index] / 2 - tree.x_arr[t_index] -
                     tree.width_arr[t_index] / 2) - (tree.width_arr[s_index] / 2 + tree.width_arr[t_index] / 2 - spacing_)
        else:
            dx = 0
            x_overlap = True

        if tree.y_arr[s_index] >= tree.y_arr[t_index] + tree.height_arr[t_index] or tree.y_arr[
                t_index] >= tree.y_arr[s_index] + tree.height_arr[s_index]:
            dy = abs(tree.y_arr[s_index] + tree.height_arr[s_index] / 2 - tree.y_arr[t_index] -
                     tree.height_arr[t_index] / 2) - (tree.height_arr[s_index] / 2 + tree.height_arr[t_index] / 2 - spacing_)
        else:
            dy = 0
            y_overlap = True

        wirelength = dx + dy
        total_wirelength += wirelength * connection_matrix[s_index][t_index]
        wlm[s_index][t_index] = wirelength
        assert not (x_overlap and y_overlap)
        assert math.isclose(wirelength, spacing_) or wirelength > spacing_

    wl = total_wirelength / (wire_count + 0.0001)
    # update the wirelength stats for normalization
    global wl_max, wl_min
    wl_max, wl_min = 0, 100
    if wl > wl_max:
        wl_max = wl
    if wl < wl_min:
        wl_min = wl
    return wl, wlm

def get_connections(connection_matrix):
    # get connection information. One time execution
    s, t = [], []
    net, wire_count = 0, 0
    n_chiplet = len(connection_matrix)
    for i in range(n_chiplet):
        for j in range(n_chiplet):
            if (i != j) and (connection_matrix[i][j] > 0):
                s.append(i)
                t.append(j)
                net += 1
                wire_count += connection_matrix[i][j]
    return net, s, t, wire_count

#矩阵元素connection_matrix[i][j]表示从chiplet i到chiplet j的连接数量

def eva_power(conn_name, ds_name, partition, optimizer) :
     # 示例调用
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

    ds = get_dataset()
    
    # 提取指定芯片图或连接关系
    bdg_all = []
    if conn_name:  # 按连接名称提取
        # 尝试按芯片名称查找
        chip_graph = None
        for vendor, chips in ds.items():
            for chip in chips:
                if hasattr(chip, 'graph') and chip.graph.get('name') == conn_name:
                    chip_graph = chip
                    break
            if chip_graph:
                break
        if chip_graph:
            bdg_all = [chip_graph]
        else:
            available = list(ds[ds_name].keys()) if ds_name in ds else []
            raise ValueError(f"找不到'{conn_name}'，可用连接: {available}")
    else:  # 使用默认全部连接
        bdg_all = ds[ds_name] if ds_name in ds else []

    vol_single = 500 * 1000
    vol_all = [vol_single] * len(bdg_all)
    
    # 将用户提供的划分格式转换为ptt_all需要的格式
    # 输入格式示例: [0, 0, 1, 1, ...] 表示节点0在组0，节点1在组0，节点2在组1...
    
    # 转换为ptt_all格式: {组号: [节点列表]}
    partition_dict = {}
    for node_id, group_id in enumerate(partition):
        if group_id not in partition_dict:
            partition_dict[group_id] = []
        partition_dict[group_id].append(node_id)
    
    # 转换为List[List[List[int]]]格式
    ptt_all = [[group for group in partition_dict.values()]]

    for idx_sys, (bdg, ptt) in enumerate(zip(bdg_all, ptt_all)):
        pm = np.zeros(shape=(len(bdg), len(bdg)), dtype=int)
        for p in ptt:
            for i in p:
                for j in p:
                    pm[i][j] = 1  # same partition, same chiplet
    power, perf = get_pp_final(bdg_all[idx_sys], ptt, optimizer)

    return power,perf
