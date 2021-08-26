from dataProcessing import PYGraphDataset, PYMake_task_input
from dataProcessing import PYGraphDatasetNaming, PYNamingMake_task_input
from dataProcessing import LRStaticGraphDataset, LRStaticMake_task_input
from dataProcessing import CNewGraphDataset, CNewMake_task_input
from dataProcessing import CGraphDataset, CMake_task_input


from models import GGNN, Tensor_GGNN_GCN

def name_to_dataset_class(name: str, args):
    name = name.lower()
    #return classname, appendix attribute, make_task_input_function, 
    if name in ["python"]:
        return PYGraphDataset, {}, PYMake_task_input
    if name in ["pythonnaming"]:
        return PYGraphDatasetNaming, {}, PYNamingMake_task_input
    if name in ["c"]:
        return CGraphDataset, {"max_node_per_graph":args.max_node_per_graph,}, CMake_task_input
    if name in ["cnew"]:
        return CNewGraphDataset, {"max_node_per_graph":args.max_node_per_graph,}, CNewMake_task_input
    if name in ["learningstatic"]:
        return LRStaticGraphDataset, {"max_node_per_graph":args.max_node_per_graph,}, LRStaticMake_task_input
    raise ValueError("Unkown dataset name '%s'" % name)

def name_to_model_class(name: str, args):
    name = name.lower()
    #return classname, appendix attribute
    if name in ["ggnn"]:
        return GGNN, {}
    if name in ["tensor_ggnn_gcn"]:
        return Tensor_GGNN_GCN, {"max_node_per_graph":args.max_node_per_graph}
    raise ValueError("Unkown model name '%s'" % name)