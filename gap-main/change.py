import torch
import dgl
import networkx as nx
from dataset.data import get_dataset
from cost_cursor.cursor import calculate_costs
from dataset.model_chip import Block, Chiplet

def convert_to_dgl_graph(nx_graph):
    """将networkx图转换为DGL图"""
    # 创建DGL图
    dgl_graph = dgl.from_networkx(nx_graph)
    
    # 从networkx图中获取芯片名称
    if 'name' in nx_graph.graph:
        # DGLGraph属性设置方式
        dgl_graph.name = nx_graph.graph['name']
        setattr(dgl_graph, 'graph_name', nx_graph.graph['name'])
    
    # 获取节点属性
    nodes = list(nx_graph.nodes(data=True))
    num_nodes = len(nodes)
    
    # 准备特征矩阵 (面积 + 5个成本特征)
    features = torch.zeros((num_nodes, 6))
    
    for idx, (_, attr) in enumerate(nodes):
        block = attr['block']
        
        # 计算成本特征 (假设每个block代表一个chiplet)
        cost_result = calculate_costs(Chiplet({block: 1}), 1, block.node)
        
        # 设置特征
        features[idx, 0] = block.area  # 面积
        features[idx, 1] = cost_result['cost_raw_chiplets']
        features[idx, 2] = cost_result['cost_defect_chiplets']
        features[idx, 3] = cost_result['cost_raw_package']
        features[idx, 4] = cost_result['cost_defect_package']
        features[idx, 5] = cost_result['cost_wasted_chiplets']
    
    # 添加特征到图
    dgl_graph.ndata['feat'] = features
    
    return dgl_graph, features

def generate_graphs_and_features():
    """
    生成所有图的graphs和features列表
    Returns:
        tuple: (graphs, features, num_nodes_list)
        - graphs: List[DGLGraph] 图列表
        - features: List[Tensor] 特征矩阵列表
        - num_nodes_list: List[int] 每个图的节点数量列表
    """
    graphs = []
    features = []
    # num_nodes_list = []
    
    # 获取所有数据集
    dataset = get_dataset()
    
    # 处理每种架构的图
    for arch_name, arch_graphs in dataset.items():
        for nx_graph in arch_graphs:
            dgl_graph, feat = convert_to_dgl_graph(nx_graph)
            graphs.append(dgl_graph)
            features.append(feat)
            # num_nodes_list.append(dgl_graph.number_of_nodes())
    
    return graphs, features

def get_graphs_and_features_by_name(target_arch_name):

    graphs = []
    features = []
    
    # 获取所有数据集
    dataset = get_dataset()
    
    # 检查目标架构是否存在
    if target_arch_name not in dataset:
        raise ValueError(f"Architecture {target_arch_name} not found in dataset. Available: {list(dataset.keys())}")
    
    # 处理目标架构的图
    for nx_graph in dataset[target_arch_name]:
        dgl_graph, feat = convert_to_dgl_graph(nx_graph)
        graphs.append(dgl_graph)
        features.append(feat)
    
    return graphs, features

def get_graphs_and_features_by_graph_name(target_graph_name, exact_match=True):
    """
    根据具体图名称获取对应的graphs和features列表
    
    Args:
        target_graph_name (str): 要匹配的图名称（如"epyc_7282"）
        exact_match (bool): 是否要求精确匹配，False时支持部分匹配
        
    Returns:
        tuple: (graphs, features)
        - graphs: List[DGLGraph] 匹配的图列表
        - features: List[Tensor] 对应的特征矩阵列表
        
    Raises:
        ValueError: 如果找不到匹配的图
    """
    graphs = []
    features = []
    
    # 获取所有数据集
    dataset = get_dataset()
    
    # 遍历所有架构和图
    for arch_graphs in dataset.values():
        for nx_graph in arch_graphs:
            # 检查图名称是否匹配
            graph_name = nx_graph.graph.get('name', '')
            if exact_match:
                match_condition = (graph_name == target_graph_name)
            else:
                match_condition = (target_graph_name in graph_name)
            
            if match_condition:
                dgl_graph, feat = convert_to_dgl_graph(nx_graph)
                graphs.append(dgl_graph)
                features.append(feat)
    
    if not graphs:
        available_names = []
        for arch_graphs in dataset.values():
            for nx_graph in arch_graphs:
                if 'name' in nx_graph.graph:
                    available_names.append(nx_graph.graph['name'])
        raise ValueError(
            f"Graph {target_graph_name} not found. Available names: {sorted(set(available_names))}"
        )
    
    return graphs, features

def ensure_single_dgl_graph(graph_input):
    """确保返回单个DGL图对象"""
    if isinstance(graph_input, list):
        if len(graph_input) == 0:
            raise ValueError("Empty graph list provided")
        # 取第一个图并确保是DGL图
        graph = graph_input[0]
        if not isinstance(graph, dgl.DGLGraph):
            raise TypeError("List contains non-DGL graph objects")
        return graph
    elif isinstance(graph_input, dgl.DGLGraph):
        return graph_input
    else:
        raise TypeError("Input must be DGLGraph or list of DGLGraphs")

def ensure_tensor_input(feature_input):
    """确保特征输入是张量"""
    if isinstance(feature_input, list):
        if len(feature_input) == 0:
            raise ValueError("Empty feature list provided")
        # 如果是张量列表，取第一个
        if isinstance(feature_input[0], torch.Tensor):
            return feature_input[0]
        # 如果是numpy数组等，转换为张量
        return torch.tensor(feature_input[0])
    elif isinstance(feature_input, torch.Tensor):
        return feature_input
    else:
        return torch.tensor(feature_input)


# num_features是指单个节点的特征维度
# 在本实现中固定为6 (面积+5个成本特征)
NUM_FEATURES = 6

if __name__ == "__main__":
    graphs, features, num_nodes_list = generate_graphs_and_features()
    print(f"Generated {len(graphs)} graphs with features")
    # 示例输出
    print("\n第一个图信息:")
    print("- 节点数量:", num_nodes_list[0])
    print("- 特征矩阵形状:", features[0].shape, f"(节点数×特征维度, 特征维度={NUM_FEATURES})")
    print("- 图结构:", graphs[0])
    
    # 统计所有图的节点数量
    print("\n节点数量统计:")
    print(f"总图数量: {len(graphs)}")
    print(f"最小节点数: {min(num_nodes_list)}")
    print(f"最大节点数: {max(num_nodes_list)}")
    print(f"平均节点数: {sum(num_nodes_list)/len(num_nodes_list):.1f}")
