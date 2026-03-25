import dgl
import torch
from eva_cost import calculate_chip_cost

def extract_node_degrees_from_dgl_graph(graph):
    node_degrees = graph.in_degrees() + graph.out_degrees()
    return node_degrees.float()

def create_adjacency_matrix(graph):
    return graph.adjacency_matrix().to_dense()

def modified_normalized_cut_loss(graph, partition_probs, epsilon=1e-5):
    '''
    gamma = partition_probs(Transpose) * node degrees
    normalized_cut_loss = (partition_probs /(element wise) gamma) * ( 1 - partition_probs)(transpose) * (element_wise) adjacency_matrix

    :param graph:
    :param partition_probs:
    :return:
    '''
    gamma = partition_probs.T @ extract_node_degrees_from_dgl_graph(graph)
    term1 = torch.div(partition_probs, gamma + epsilon)
    term2 = (1 - partition_probs).T
    term3 = create_adjacency_matrix(graph)
    loss = (term1 @ term2) * term3
    return loss.sum()

def modified_normalized_cut_loss_2(graph, partition_probs):
    node_degrees = extract_node_degrees_from_dgl_graph(graph)
    adjacency_matrix = create_adjacency_matrix(graph)
    gamma = partition_probs.T @ node_degrees
    term1 = torch.div(partition_probs, gamma)
    term2 = (1 - partition_probs).T
    term1_term2 = term1 @ term2
    loss_matrix = term1_term2 * adjacency_matrix
    loss = loss_matrix.sum()
    return loss

def modified_balanced_partition_error(partition_probs):
    '''
    reduce-sum(1.T * partition_probs - n/g)^2
    :param partition_probs: nxg tensor
    :param n: given the number of nodes in the graph
    :param g: number of partitions
    :return:
    '''
    n = partition_probs.size(0)
    g = partition_probs.size(1)
    ones_vector = torch.ones(n, 1)
    balance_error = (((ones_vector.T @ partition_probs) - (n / g)) ** 2).sum()
    #print((ones_vector.T @ partition_probs), "Shape: " ,(ones_vector.T @ partition_probs).shape)
    return balance_error

def modified_areabalanced_partition_error(graph, partition_probs):
    """
    计算基于面积的分区平衡误差，并添加对小分区的惩罚
    Args:
        graph: DGL图对象
        partition_probs: 分区概率矩阵 [n_nodes, n_partitions]
        min_area_penalty: 对小分区的惩罚系数
        epsilon: 避免除零的小常数
    Returns:
        平衡误差 + 小分区惩罚
    """
    # 获取节点面积（features的第一项）
    areas = graph.ndata['feat'][:, 0].unsqueeze(1)  # [n,1]
    total_area = areas.sum()
    g = partition_probs.size(1)
    
    #软分区
    partition_areas = (areas.T @ partition_probs)
    balance_error = ((partition_areas - (total_area / g)) ** 2).sum()     # 原始平衡误差（各分区面积与平均面积的平方差和）
    
    # 添加存在性惩罚
    penalty_weight=1e4
    eps=1e-5
    partition_probs_sum = partition_probs.sum(dim=0)  # 各分区概率总和 [g]
    min_prob = torch.min(partition_probs_sum)  # 找出概率总和最小的分区
    existence_penalty = penalty_weight * torch.sigmoid(-min_prob/eps)  # 平滑惩罚
    
    return balance_error + existence_penalty


def modified_loss(graph, partition_probs, alpha, beta, epoch=None):
    '''

    :param graph:
    :param partition_probs:
    :return:
    '''

    cut_loss = modified_normalized_cut_loss(graph, partition_probs)
    #balance_error = modified_balanced_partition_error(partition_probs)
    balance_error = modified_areabalanced_partition_error(graph, partition_probs)
    cost_loss = modified_cost_loss(graph, partition_probs)

    if epoch is not None and epoch % 1 == 0:
        print(f'Epoch: {epoch}, Cut Loss: {cut_loss}, Balance Error: {alpha*balance_error}, Cost Error: {beta*cost_loss}, Total Loss: {cut_loss + alpha*balance_error + beta*cost_loss}')
    return cut_loss + alpha*balance_error +  beta*cost_loss

# Evaluation Metrics
def edge_cut_ratio(graph, partition_probs):
    with graph.local_scope():
        graph.ndata['p'] = partition_probs
        graph.apply_edges(dgl.function.u_mul_v('p', 'p', 'e')) #
        edge_cut = graph.edata['e'].sum().item()
    total_edges = graph.num_edges()
    return edge_cut / total_edges

def balancedness(partition_probs, n, g):
    partition_sizes = partition_probs.sum(dim=0)
    ideal_size = n/g
    balance_error = ((partition_sizes - ideal_size) ** 2).mean().item()
    balance_error = balance_error ** 0.5
    balance_error /= ideal_size
    return 1-balance_error

def evaluate(graph, partition_probs):
    n = graph.number_of_nodes()
    g = partition_probs.size(1)
    cut_ratio = edge_cut_ratio(graph, partition_probs)
    balance_score = balancedness(partition_probs, n, g)
    return cut_ratio, balance_score


# def modified_cost_loss(graph, partition_probs):
#     chip_name = graph.name
#     cost_dict = calculate_chip_cost(chip_name, graph, partition_probs)
#     total_cost = cost_dict.get('re_cost', {}).get('total', torch.tensor(0.0).to(partition_probs.device)) \
#                  if isinstance(cost_dict, dict) else cost_dict
    
#     # 动态缩放因子（基于初始成本）
#     if not hasattr(modified_cost_loss, 'init_cost'):
#         modified_cost_loss.init_cost = total_cost.detach()
#     scale_factor = 10 / (modified_cost_loss.init_cost + 1e-5)
    
#     return total_cost * scale_factor if isinstance(total_cost, torch.Tensor) \
#            else torch.tensor(total_cost).to(partition_probs.device) * scale_factor
#     # # 确保返回的是与partition_probs相同设备和类型的张量
#     # return total_cost if isinstance(total_cost, torch.Tensor) \
#     #        else torch.tensor(total_cost).to(partition_probs.device)

def modified_cost_loss(graph, partition_probs):
    chip_name = graph.name
    cost_dict = calculate_chip_cost(chip_name, graph, partition_probs)
    total_cost = cost_dict.get('re_cost', {}).get('total', torch.tensor(0.0).to(partition_probs.device)) \
                 if isinstance(cost_dict, dict) else cost_dict
    
    # 确保total_cost是tensor
    if not isinstance(total_cost, torch.Tensor):
        total_cost = torch.tensor(total_cost).to(partition_probs.device)
    # 对数变换增强梯度 (+1避免log(0))
    log_cost = torch.log(total_cost + 1.0)
    # 动态缩放因子（基于初始成本）
    if not hasattr(modified_cost_loss, 'init_cost'):
        modified_cost_loss.init_cost = log_cost.detach()
    # 计算缩放因子（初始成本越大，缩放越小）
    scale_factor = 10.0 / (modified_cost_loss.init_cost + 1e-5)
    # 应用缩放并返回
    return log_cost * scale_factor


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

def calculate_area(graph, partition_probs):

    node_areas = graph.ndata['feat'][:, 0]
    n_partitions = partition_probs.size(1)
    
    partitions = calculate_partition(graph, partition_probs)
    # 计算各分区总面积
    partition_areas = torch.zeros(n_partitions, device=partition_probs.device)
    partition_areas.scatter_add_(0, partitions, node_areas)
    
    return {i: area.item() for i, area in enumerate(partition_areas)}


