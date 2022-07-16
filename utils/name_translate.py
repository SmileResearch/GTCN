from enum import Enum
import json
from multiprocessing import cpu_count
import torch.nn as nn


# 本地库
from models import GGNN, residual_graph_attention, GNN_FiLM, Edge_Conv, MTFF_Co_Attention, Tensor_GCN
from models import Transformer_GCN, Relational_GCN, Deep_GCN
from dataProcessing import CSharpStaticGraphDatasetGenerator, PythonStaticGraphDatasetGenerator
from tasks import VarmisuseOutputLayer, VarnamingOutputLayer
from .model_metrics import cal_metrics


class DataFold(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


concat_singal = "_- ,/\\"


def name_to_dataloader(name: str, path: str, data_fold: DataFold, args, num_workers=int(cpu_count() / 2)):  # 返回类型为对象
    """将字符串转为数据预处理对象

    Args:
        name (str): 数据预处理的类名
        path (str): 数据目录地址
        args (_type_): 可能用到的参数

    Raises:
        ValueError: 类名不存在
    """
    name = name.lower()
    name = name.replace(concat_singal, "")
    if name in ["csharp"]:
        datasetCls = CSharpStaticGraphDatasetGenerator
    elif name in ["python", "py150", "py"]:
        datasetCls = PythonStaticGraphDatasetGenerator
    else:
        raise ValueError("Unkown dataset name '%s'" % name)

    # 导入字典
    with open(args.value_dict_dir, 'r') as f:
        value_dict = json.load(f)
    with open(args.type_dict_dir, 'r') as f:
        type_dict = json.load(f)

    # 返回数据库对象
    return datasetCls(
        path,
        value_dict,
        type_dict,
        num_edge_types=args.num_edge_types,
        batch_size=args.batch_size,
        shuffle=True if data_fold == DataFold.TRAIN else False,
        graph_node_max_num_chars=args.graph_node_max_num_chars,
        max_graph=args.max_graph,
        max_variable_candidates=args.max_variable_candidates,
        max_node_per_graph=args.max_node_per_graph,
        num_workers=num_workers,
        device=args.device,
        slice_edge_type=args.slice_edge_type,
        slot_singal=args.slot_singal,
    )



def name_to_model(name: str, args, **kwargs):
    """将字符串转为模型并返回。

    Args:
        name (str): 模型名称。推荐小写。
        args (_type_): 可能会用到的参数
        **kwargs: 可能需要传入的其他字典型参数。

    Raises:
        ValueError: 类名不存在
    """
    name = name.lower()
    name = name.replace(concat_singal, "")
    if name in ["ggnn", "graphgatedneuralnetwork"]:
        return GGNN(num_edge_types=args.num_edge_types,
                    in_features=args.graph_node_max_num_chars,
                    out_features=args.out_features,
                    embedding_out_features=args.h_features,
                    embedding_num_classes=70,
                    dropout=args.dropout_rate,
                    device=args.device)
    elif name in ["resgagn"]:
        return residual_graph_attention(num_edge_types=args.num_edge_types,
                       in_features=args.graph_node_max_num_chars,
                       out_features=args.out_features,
                       embedding_out_features=args.h_features,
                       embedding_num_classes=70,
                       dropout=args.dropout_rate,
                       max_node_per_graph=args.max_node_per_graph,
                       device=args.device)
    elif name in ["gnn_film", "gnnfilm"]:
        return GNN_FiLM(num_edge_types=args.num_edge_types,
                        in_features=args.graph_node_max_num_chars,
                        out_features=args.out_features,
                        embedding_out_features=args.h_features,
                        embedding_num_classes=70,
                        dropout=args.dropout_rate,
                        device=args.device)
    elif name in ["edge_conv", "edgeconv"]:
        return Edge_Conv(num_edge_types=args.num_edge_types,
                         in_features=args.graph_node_max_num_chars,
                         out_features=args.out_features,
                         embedding_out_features=args.h_features,
                         embedding_num_classes=70,
                         dropout=args.dropout_rate,
                         device=args.device)
    elif name in ["mtff_co_attention", "mtff_module", "mtff_model", "mtff"]:
        return MTFF_Co_Attention(
            feature_models=kwargs["feature_models"],
            in_features=args.graph_node_max_num_chars,
            out_features=args.out_features,
            embedding_out_features=args.h_features,
            embedding_num_classes=70,
            max_node_per_graph=args.max_node_per_graph,
            device=args.device
        )
    elif name in ["tensorgcn", "tensor_gcn"]:
        return Tensor_GCN(
            num_edge_types=args.num_edge_types,
            in_features=args.graph_node_max_num_chars,
            out_features=args.out_features,
            embedding_out_features=args.h_features,
            embedding_num_classes=70,
            dropout=args.dropout_rate,
            max_node_per_graph=args.max_node_per_graph,
            device=args.device
        )
    elif name in ["transformer_gcn", "transformergcn", "tgcn"]:
        return Transformer_GCN(num_edge_types=args.num_edge_types,
                    in_features=args.graph_node_max_num_chars,
                    out_features=args.out_features,
                    embedding_out_features=args.h_features,
                    embedding_num_classes=70,
                    dropout=args.dropout_rate,
                    device=args.device)
    elif name in ["realtional_gcn", "realtionalgcn", "rgcn"]:
        return Relational_GCN(num_edge_types=args.num_edge_types,
                    in_features=args.graph_node_max_num_chars,
                    out_features=args.out_features,
                    embedding_out_features=args.h_features,
                    embedding_num_classes=70,
                    dropout=args.dropout_rate,
                    device=args.device)
    elif name in ["deep_gcn", "deepgcn", "dgcn"]:
        return Deep_GCN(num_edge_types=args.num_edge_types,
                    in_features=args.graph_node_max_num_chars,
                    out_features=args.out_features,
                    embedding_out_features=args.h_features,
                    embedding_num_classes=70,
                    dropout=args.dropout_rate,
                    device=args.device)
    else:
        raise ValueError("Unkown model name '%s'" % name)


def name_to_output_model(name: str, args):
    """将字符串转为输出模型并返回。

    Args:
        name (str): 输出模型名称。推荐小写。
        args (_type_): 可能会用到的参数

    Raises:
        ValueError: 类名不存在
    """
    with open(args.value_dict_dir, 'r') as f:
        value_dict = json.load(f)
    with open(args.type_dict_dir, 'r') as f:
        type_dict = json.load(f)
    
    name = name.lower()
    name = name.replace(concat_singal, "")
    if name in ["vm", "varmisuse", "variablemisuse"]:
        return VarmisuseOutputLayer(out_features=args.out_features,
                                    max_variable_candidates=args.max_variable_candidates,
                                    criterion=nn.CrossEntropyLoss(),
                                    metrics=cal_metrics,
                                    device=args.device)
    elif name in ["cc", "varnaming",  "codecompletion" ]:
        # TODO: 待补充
        return VarnamingOutputLayer(out_features=args.out_features,
                                    classifier_nums=len(value_dict)+1,
                                    criterion=nn.CrossEntropyLoss(),
                                    metrics=cal_metrics,
                                    device=args.device
                                    )
    else:
        raise ValueError("Unkown output model name '%s'" % name)