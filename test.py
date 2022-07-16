from dataProcessing import PYGraphDataset, LRStaticGraphDataset
from torch_geometric.data import DataLoader
import json
if __name__ == '__main__':
    from torch_geometric.data import DataLoader
    value_dict_dir = "vocab_dict/python_terminal_dict_1k_value.json"
    type_dict_dir = "vocab_dict/python_terminal_dict_1k_type.json"
    with open(value_dict_dir, 'r') as f:
        value_dict = json.load(f)
    with open(type_dict_dir, 'r') as f:
        type_dict = json.load(f)
   
    # gd = PYGraphDataset("data/pytemp/validate_data", value_dict=value_dict, type_dict=type_dict, max_node_per_graph=512, max_graph=10000)
    # for idx, example in enumerate(gd[:2]):
    #     print("*** Example ***")
    #     print("idx: {}".format(idx))
    #     print("label: {}".format(example.label))
    #     #print("x: {}".format(example.x))
    #     print("x shape: {}".format(example.x.shape))
    #     print("slot_id: {}".format(example.slot_id))
    #     print("candidate_ids: {}".format(example.candidate_ids))
    #     print("candidate_ids len: {}".format(len(example.candidate_ids)))
    #     print("candidate_masks: {}".format(example.candidate_masks))
    #     #print("ast: {}".format(example.ast))
    #     print("ast shape: {}".format(example.ast_index.shape))
    #     #print("ncs: {}".format(example.ncs))
    #     print("ncs shape: {}".format(example.ncs_index.shape))
    #     #print("comesFrom: {}".format(example.comesFrom))
    #     print("comesFrom shape: {}".format(example.comesFrom_index.shape))
    #     #print("computedFrom: {}".format(example.computedFrom))
    #     print("computedFrom shape: {}".format(example.computedFrom_index.shape))
    # dl = DataLoader(gd, batch_size=32)
    # for d in dl:
    #     print("d.x.shape", d.x.shape)
    #     print("d.comesFrom_index.shape", d.comesFrom_index.shape)
    #     print("d.ast_index.shape", d.ast_index.shape)


    gd = LRStaticGraphDataset("data/lrtemp/train_data", value_dict=value_dict, type_dict=type_dict, max_node_per_graph=10000, max_variable_candidates=10)
    for idx, example in enumerate(gd[:10]):
        print("*** Example ***")
        print("example.__slot_node_name__: {}".format(example.slot_node_name))
        print("idx: {}".format(idx))
        print("label: {}".format(example.label))
        #print("x: {}".format(example.x))
        print("x shape: {}".format(example.x.shape))
        print("slot_id: {}".format(example.slot_id))
        print("candidate_ids: {}".format(example.candidate_ids))
        print("candidate_ids len: {}".format(len(example.candidate_ids)))
        print("candidate_masks: {}".format(example.candidate_masks))
        #print("ast: {}".format(example.ast))
        print("Child_index shape: {}".format(example.Child_index.shape))
        #print("ncs: {}".format(example.ncs))
        print("NextToken_index shape: {}".format(example.NextToken_index.shape))
        #print("comesFrom: {}".format(example.comesFrom))
        print("LastUse_index shape: {}".format(example.LastUse_index.shape))
        #print("computedFrom: {}".format(example.computedFrom))
        print("LastWrite_index shape: {}".format(example.LastWrite_index.shape))
    dl = DataLoader(gd, batch_size=32)
    print("==========dataloader")
    for d in dl:
        single_data = d.to_data_list()[0]
        print("single_data.__slot_node_name__", single_data.slot_node_name)
        print("d.x.shape", d.x.shape)
        print("d.Child_index.shape", d.Child_index.shape)
        print("d.NextToken_index.shape", d.NextToken_index.shape)