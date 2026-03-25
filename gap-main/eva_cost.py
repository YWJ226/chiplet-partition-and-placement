import numpy as np
from dataset.data import get_dataset
from dataset.model_chip import Chiplet, Block, SI
import torch

def graph_to_chip_name(graph):
    """从graph对象提取芯片名称"""
    if hasattr(graph, 'name'):
        return graph.name
    elif hasattr(graph, 'graph_name'):
        return getattr(graph, 'graph_name')
    else:
        raise ValueError("Graph对象不包含芯片名称信息")

def calculate_chip_cost(chip_name, graph_data, cluster_labels):

    # 获取芯片连接关系
    datasets = get_dataset()
    chip_graph = None
    for vendor, chips in datasets.items():
        for chip in chips:
            if chip.graph['name'] == chip_name:
                chip_graph = chip
                break
        if chip_graph:
            break
    
    if not chip_graph:
        raise ValueError(f"未找到芯片: {chip_name}")

    # 获取节点属性
    nodes = list(chip_graph.nodes(data=True))
    if len(nodes) != len(cluster_labels):
        raise ValueError(f"标签数量({len(cluster_labels)})与芯片节点数({len(nodes)})不匹配")

    # 将概率转换为硬标签
    # if hasattr(cluster_labels, 'detach'):
    #     cluster_labels = cluster_labels.detach()
    # if hasattr(cluster_labels, 'argmax'):
    #     cluster_labels = cluster_labels.argmax(dim=1).cpu().numpy()
    cluster_labels = calculate_partition(graph_data, cluster_labels)
    unique_clusters = np.unique(cluster_labels)
 
    cost_details = {}
    
    # 收集并合并相同chiplets
    chiplets = {}
    for cluster_id in unique_clusters:
        cluster_nodes = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        
        # 为cluster中每个node创建block
        blocks = {}
        for node_idx in cluster_nodes:
            node_block = nodes[node_idx][1]['block']
            block = Block(node_block.name, node_block.area, node_block.node)
            if block in blocks:
                blocks[block] += 1
            else:
                blocks[block] = 1
        
        # 创建chiplet
        chiplet = Chiplet(blocks)
        
        # 查找相同chiplet
        found = None
        for existing in chiplets:
            if existing == chiplet:  # 使用Chiplet的__eq__方法比较
                found = existing
                break
                
        if found:
            chiplets[found] += 1
        else:
            chiplets[chiplet] = 1
    
    try:
        # 使用SI类计算整个芯片成本
        si = SI(chiplets)
        # 计算RE成本
        re_costs = si.RE()
        cost_raw_chiplets, cost_defect_chiplets, cost_raw_package, cost_defect_package, cost_wasted_chiplets = re_costs
        re_total = sum(re_costs)
        # 计算NRE成本
        nre_cost = si.NRE()
        
        # 记录成本
        cost_details = {
            're_cost': {
                'total': re_total,
                'raw_chiplets': cost_raw_chiplets,
                'defect_chiplets': cost_defect_chiplets,
                'raw_package': cost_raw_package,
                'defect_package': cost_defect_package,
                'wasted_chiplets': cost_wasted_chiplets
            },
            'nre_cost': nre_cost,
            'total_cost': re_total + nre_cost
        }
    except Exception as e:
        raise ValueError(f"计算芯片成本失败: {str(e)}")
    return cost_details

# def calculate_partition(graph, partition_probs):
#     node_areas = graph.ndata['feat'][:, 0]
#     n_nodes = partition_probs.size(0)
#     n_partitions = partition_probs.size(1)
    
#     base_k = n_nodes // n_partitions
#     remainder = n_nodes % n_partitions
    
#     partitions = torch.full((n_nodes,), -1, device=partition_probs.device)
    
#     # 第一阶段：硬分配基础节点
#     for p in range(n_partitions):
#         # 获取该分区概率最高的base_k个节点
#         weights = partition_probs[:, p]  # 可以加* node_areas如果需要考虑面积
#         topk_indices = torch.topk(weights, base_k).indices
        
#         for idx in topk_indices:
#             if partitions[idx] == -1:
#                 partitions[idx] = p
#             else:
#                 if partition_probs[idx, p] > partition_probs[idx, partitions[idx]]:
#                     partitions[idx] = p
    
#     # 第二阶段：分配剩余节点（包括余数节点）
#     unassigned = (partitions == -1).nonzero().squeeze()
#     if unassigned.numel() > 0:
#         # 对每个未分配节点，选择概率最高的分区
#         _, max_part = torch.max(partition_probs[unassigned], dim=1)
        
#         # 确保分区不超过容量限制
#         partition_counts = torch.zeros(n_partitions, dtype=torch.int32)
#         for p in range(n_partitions):
#             partition_counts[p] = (partitions == p).sum()
        
#         max_capacity = base_k + 1  # 每个分区最多base_k+1个节点
        
#         for i, p in zip(unassigned, max_part):
#             if partition_counts[p] < max_capacity:
#                 partitions[i] = p
#                 partition_counts[p] += 1
#             else:
#                 # 如果首选分区已满，选择次优分区
#                 probs = partition_probs[i]
#                 sorted_partitions = torch.argsort(probs, descending=True)
#                 for sp in sorted_partitions:
#                     if partition_counts[sp] < max_capacity:
#                         partitions[i] = sp
#                         partition_counts[sp] += 1
#                         break
    
#     return partitions

def calculate_partition(graph, partition_probs):
    node_areas = graph.ndata['feat'][:, 0]
    n_nodes = partition_probs.size(0)
    n_partitions = partition_probs.size(1)
    
    base_k = n_nodes // n_partitions
    remainder = n_nodes % n_partitions
    
    partitions = torch.full((n_nodes,), -1, device=partition_probs.device)
    
    # 第一阶段：硬分配基础节点
    for p in range(n_partitions):
        weights = partition_probs[:, p]
        topk_indices = torch.topk(weights, base_k).indices
        
        for idx in topk_indices:
            if partitions[idx] == -1:
                partitions[idx] = p
            else:
                if partition_probs[idx, p] > partition_probs[idx, partitions[idx]]:
                    partitions[idx] = p
    
    # 第二阶段：分配剩余节点（包括余数节点）
    unassigned = (partitions == -1).nonzero()
    if unassigned.numel() > 0:
        # 确保unassigned保持2D形状 [N,1]
        if unassigned.dim() == 1:
            unassigned = unassigned.unsqueeze(1)
        
        # 获取每个未分配节点的概率分布，保持2D形状
        probs = partition_probs[unassigned.squeeze(1)]  # [N, n_partitions]
        if probs.dim() == 1:  # 处理单节点情况
            probs = probs.unsqueeze(0)
        
        # 获取每个节点的最佳分区
        _, max_part = torch.max(probs, dim=1)
        
        # 确保分区不超过容量限制
        partition_counts = torch.zeros(n_partitions, dtype=torch.int32)
        for p in range(n_partitions):
            partition_counts[p] = (partitions == p).sum()
        
        max_capacity = base_k + 1
        
        for i, p in zip(unassigned, max_part):
            if partition_counts[p] < max_capacity:
                partitions[i] = p
                partition_counts[p] += 1
            else:
                # 如果首选分区已满，选择次优分区
                probs_i = partition_probs[i.squeeze()]
                sorted_partitions = torch.argsort(probs_i, descending=True)
                for sp in sorted_partitions:
                    if partition_counts[sp] < max_capacity:
                        partitions[i] = sp
                        partition_counts[sp] += 1
                        break
    
    return partitions



def save_cost_results(results, output_file):
    """保存成本结果到文件"""
    with open(output_file, 'w') as f:
        f.write("Cost Category\tCost Type\tValue\n")
        # 输出RE成本
        f.write(f"RE\tRawChiplets\t{results['re_cost']['raw_chiplets']:.2f}\n")
        f.write(f"RE\tDefectChiplets\t{results['re_cost']['defect_chiplets']:.2f}\n")
        f.write(f"RE\tRawPackage\t{results['re_cost']['raw_package']:.2f}\n")
        f.write(f"RE\tDefectPackage\t{results['re_cost']['defect_package']:.2f}\n")
        f.write(f"RE\tWastedChiplets\t{results['re_cost']['wasted_chiplets']:.2f}\n")
        f.write(f"RE\tTotal\t{results['re_cost']['total']:.2f}\n")
        # 输出NRE成本
        f.write(f"NRE\tNRE_Cost\t{results['nre_cost']:.2f}\n")
        # 输出总成本
        f.write(f"Total\tTotalCost\t{results['total_cost']:.2f}\n")

if __name__ == "__main__":
    # 示例用法
    chip_name = "epyc_7282"  # 可从daegc.py获取实际芯片名
    # y_pred = np.array([11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # 示例聚类结果，实际从DAEGC获取
    y_pred = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 0, 1, 2, 3, 4, 5, 6, 7])  # 示例聚类结果，实际从DAEGC获取
    try:
        cost_results = calculate_chip_cost(chip_name, y_pred)
        save_cost_results(cost_results, "chip_cost_results.txt")
        print(f"成本计算完成,结果已保存到chip_cost_results.txt")
        print(f"总成本: {cost_results['total_cost']:.2f}")
    except Exception as e:
        print(f"成本计算失败: {str(e)}")
