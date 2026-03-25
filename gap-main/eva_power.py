from dataset.evaluation import eval_ev_ppc
from dataset.data import get_dataset

def main(partition, ds_name="AMD", conn_name=None):
    """
    参数:
    partition: 芯片划分方案，格式为[0,0,1,1,...] (必需)
    ds_name: 数据集名称 (如 "AMD", "Nvidia")，默认为"AMD"
    conn_name: 要提取的特定连接关系名称(可选)
    """
    # 获取数据集
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
    
    # 设置参数
    if vendor != "Nvidia":
        indi_T_end, holi_T_end = 0.2, 0.5
    else:
        indi_T_end, holi_T_end = 0.1, 0.1
    
    params = {
        "num_cpu": 32,
        "indi_pnum": 2,
        "indi_max_try": 10,
        "indi_T_start": 1,
        "indi_T_end": indi_T_end,
        "indi_alpha": 0.95,
        "holi_pnum": 24,
        "holi_num_init_sample": 20,
        "holi_max_try": 10,
        "holi_T_start": 1,
        "holi_T_end": holi_T_end,
        "holi_alpha": 0.95,
        "type_pkg": "SI",
        "w_power": 0,
        "w_perf": 0,
        "w_cost": 1
    }

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
    
    ev, ppc = eval_ev_ppc(
        bdg_all=bdg_all,
        vol_all=vol_all,
        w_power=params["w_power"],
        w_perf=params["w_perf"],
        w_cost=params["w_cost"],
        type_pkg=params["type_pkg"],
        ptt_all=ptt_all
    )
    
    return ev, ppc

if __name__ == "__main__":
    # 示例调用
    partition = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 
                 8, 8, 8, 8, 8, 8, 8, 8, 0, 1, 2, 3, 4, 5, 6, 7]
    conn_name = "epyc_7282"  # 连接关系名称

    # 使用默认ds_name="AMD"
    ev, ppc = main(partition, conn_name=conn_name)
    print(f"Evaluation results for AMD:")
    
    # 或指定ds_name
    # ev, ppc = main(partition, ds_name="Nvidia", conn_name=conn_name)
    # print(f"Evaluation results for Nvidia:")
    print(f"EV: {ev}")
    print(f"PPC: {ppc}")
