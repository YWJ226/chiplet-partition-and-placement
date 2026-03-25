import torch
#torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()
import dgl
import torch.optim as optim
from model import GAPModel, weights_init
from utils import evaluate, modified_loss,calculate_area,calculate_partition
import networkx as nx
import random
import numpy as np
from eva_cost import calculate_chip_cost
#from change import generate_graphs_and_features,get_graphs_and_features_by_name,get_graphs_and_features_by_graph_name
import change
import os
import matplotlib.pyplot as plt
import json

def generate_erdos_renyi_graph(num_nodes, prob):
    g = nx.erdos_renyi_graph(num_nodes, prob, directed=False)
    return dgl.from_networkx(g)


def train(model, graphs=[], features=[], optimizer=None, num_epochs=1000):
    # 初始化成本记录
    cost_history = {f"graph_{i}": [] for i in range(len(graphs))}
    
    for epoch in range(num_epochs):
        for graph, index in zip(graphs, range(len(graphs))):
            model.train()
            partition_probs = model(graph, features[index])
            loss = modified_loss(graph, partition_probs, alpha=1e-2, beta = 5e-1, epoch=epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每5个epoch记录一次成本
            if epoch % 5 == 0:
                # chip_name = graph_to_chip_name(graph)
                chip_name = graph.name
                cost_dict = calculate_chip_cost(chip_name, graph, partition_probs)
                # Extract total cost from dictionary if it exists, otherwise use 0
                total_cost = cost_dict.get('re_cost', {}).get('total',0) if isinstance(cost_dict, dict) else cost_dict
                cost_history[f"graph_{index}"].append((epoch, total_cost))
            
   # print(f'raw_chiplets: cost_raw_chiplets,defect_chiplets: cost_defect_chiplets,raw_package: cost_raw_package,defect_package: cost_defect_package,wasted_chiplets: cost_wasted_chiplets')
    # 输出详细的成本信息
    if isinstance(cost_dict, dict) and 're_cost' in cost_dict:
        re_cost = cost_dict['re_cost']
        print(f"\n详细成本信息:")
        print(f"raw_chiplets: {re_cost.get('raw_chiplets', 0)}")
        print(f"defect_chiplets: {re_cost.get('defect_chiplets', 0)}")
        print(f"raw_package: {re_cost.get('raw_package', 0)}")
        print(f"defect_package: {re_cost.get('defect_package', 0)}")
        print(f"wasted_chiplets: {re_cost.get('wasted_chiplets', 0)}")
        print(f"总成本: {total_cost}")
    else:
        print(f"总成本: {total_cost}")

    print('Training complete!')
    
    # 绘制并保存成本曲线
    os.makedirs('fig', exist_ok=True)
    for graph_idx in range(len(graphs)):
        epochs, costs = zip(*cost_history[f"graph_{graph_idx}"])
        chip_name = graphs[graph_idx].name
        plt.figure()
        plt.plot(epochs, costs)
        plt.title(f"Cost Trend - Graph {chip_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.savefig(f'fig/graph_{chip_name}.png')
        plt.close()

    for graph, index in zip(graphs, range(len(graphs))):
        with torch.no_grad():
            train_partition_probs = model(graph, features[index])
        test_cut_ratio, test_balance_score = evaluate(graph, train_partition_probs)
        print(f"Train Edge Cut Ratio: {test_cut_ratio}")
        print(f"Train Balancedness: {test_balance_score}")
        
        # 输出分区面积信息
        partition_areas = calculate_area(graph, train_partition_probs)
        print("\nPartition Areas:")
        for part, area in partition_areas.items():
            print(f"Partition {part}: {area}")
        
        # partition = torch.argmax(train_partition_probs, dim=1).tolist()
        partition = calculate_partition(graph, train_partition_probs).tolist()
        print("\nPartition Assignment:")
        print("partition = [", end="")
        for i, p in enumerate(partition):
            print(p, end="")
            if i != len(partition)-1:
                print(", ", end="")
            if (i+1) % 8 == 0 and i != len(partition)-1:  # 每8个换行
                print("\n" + " " * 15, end="")
        print("]")



    return model

def main():


    seed = 41  # 可以设置为任意整数
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    num_features = 6
    num_partitions = 10
    hidden_features = 48 #48

    with open('partition_input.json', 'r') as f:
        input_data = json.load(f)

    results = {}

    # graphs = []
    # features = []
    # #graphs, features = generate_graphs_and_features()
    # #graphs, features = get_graphs_and_features_by_name('AMD')
    # graphs, features = change.get_graphs_and_features_by_graph_name('ga102')

    model = GAPModel(num_features, hidden_features, num_partitions)
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=7.5e-5, weight_decay=1e-4)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    num_epochs = 200

    #model = train(model, graphs, features, optimizer, num_epochs)

    # print('Test Results')
    # testname = 'tiangang'
    # test_graphs, test_features = change.get_graphs_and_features_by_graph_name(testname)
    # test_graph = change.ensure_single_dgl_graph(test_graphs)
    # test_feature = change.ensure_tensor_input(test_features)
    # model.eval()
    # with torch.no_grad():
    #      test_partition_probs = model(test_graph, test_feature)

    # test_cost_dict = calculate_chip_cost(testname, test_partition_probs)
    # test_total_cost = test_cost_dict.get('re_cost', {}).get('total',0) if isinstance(test_cost_dict, dict) else test_cost_dict
    # print(test_total_cost)
    # # Evaluate on the test graph
    # test_cut_ratio, test_balance_score = evaluate(test_graph, test_partition_probs)
    # print(f"Test Edge Cut Ratio: {test_cut_ratio}")
    # print(f"Test Balancedness: {test_balance_score}")
    for chip_name in input_data['chips']:

        family = detect_chip_family(chip_name)
        print(f"\nProcessing chip: {chip_name}")
        graphs, features = change.get_graphs_and_features_by_graph_name(chip_name)
        
        # 训练模型
        trained_model = train(model, graphs, features, optimizer, num_epochs)
        
        # 获取分区结果
        with torch.no_grad():
            train_partition_probs = trained_model(graphs[0], features[0])
            partition_areas = calculate_area(graphs[0], train_partition_probs)
            partition = calculate_partition(graphs[0], train_partition_probs).tolist()
            
            # 保存结果
            results[chip_name] = {
                "family": family,
                "partition_areas": partition_areas,
                "partition_assignment": partition
            }
    
    # 写入输出JSON
    with open('partition_output.json', 'w') as f:
        json.dump({"results": results}, f, indent=2)


def detect_chip_family(chip_name):
    """根据芯片名称自动检测所属系列"""
    # AMD系列特征
    if any(x in chip_name for x in ['epyc', 'ryzen']):
        return 'AMD'
    # Intel系列特征
    elif 'xeon' in chip_name:
        return 'Intel'
    # Rockchip系列特征
    elif chip_name.startswith('rk'):
        return 'Rockchip'
    # HiSilicon系列特征
    elif any(x in chip_name for x in ['kunpeng', 'tiangang', 'ascend']):
        return 'HiSilicon'
    # Nvidia系列特征
    elif chip_name.startswith('ga'):
        return 'Nvidia'
    # Allwinner系列特征
    elif any(x in chip_name for x in ['t5', 'a133', 'r818', 'a63', 'r328', 'h616', 'h80', 'a83t', 'h6', 'a50']):
        return 'Allwinner'
    else:
        return 'Other'


if __name__ == '__main__':

    main()
