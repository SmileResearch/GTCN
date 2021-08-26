from torch._C import device
import argparse
from enum import Enum
import time
import os.path as osp
from os import getpid
import json
from typing import Any, Dict, Optional, Tuple, List, Iterable
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
#from dataProcessing import CGraphDataset, PYGraphDataset, LRGraphDataset
import numpy as np
from utils import cal_metrics, top_5, cal_early_stopping_metric, pretty_print_epoch_task_metrics
from multiprocessing import cpu_count
from models import name_to_dataset_class, name_to_model_class

class DataFold(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


class CodeCompletion:
    @classmethod
    def default_args(cls):
        parser = argparse.ArgumentParser(description="Tensor_GGNN_GCN")
        # 注意检查label，检查数据集，检查train_data_dir
        parser.add_argument('--dataset_name',
                            type=str,
                            default="learningstatic",
                            help='the name of the dataset. optional:[python, learning]')
        parser.add_argument('--backbone_model',
                            type=str,
                            default="tensor_ggnn_gcn",
                            help='the backbone model of the model. optional:[dgap, ggnn]')
        parser.add_argument('--output_model',
                            type=str,
                            default="learning",
                            help='the output model of the model optional:[python, learning]')
        parser.add_argument('--optimizer', type=str, default="Adam", help='')
        parser.add_argument('--lr', type=float, default=0.001, help='')
        parser.add_argument('--lr_deduce_per_epoch', type=int, default=10, help='')
        parser.add_argument('--max_epochs', type=int, default=1500, help='')
        parser.add_argument('--cur_epoch', type=int, default=1, help='')
        parser.add_argument('--max_variable_candidates', type=int, default=5, help='')
        parser.add_argument('--batch_size', type=int, default=64, help='')
        parser.add_argument('--result_dir',
                            type=str,
                            default="trained_models/",
                            help='')
        parser.add_argument('--dropout_rate', type=float, default=0., help='keep_prob = 1-dropout_rate')
        parser.add_argument('--load_model_file',
                            type=str,
                            default=None,
                            help='')
        #parser.add_argument('--in_features', type=int, default=64, help='')  # in_features 为embedding中的graph_node_max_num_chars
        parser.add_argument('--h_features', type=int, default=64, help='')
        parser.add_argument('--out_features', type=int, default=64, help='')
        parser.add_argument('--graph_node_max_num_chars',
                            type=int,
                            default=19,
                            help='')
        parser.add_argument('--max_node_per_graph',
                            type=int,
                            default=50,
                            help='')
        parser.add_argument('--device', type=str, default="cuda", help='')
        parser.add_argument(
            '--value_dict_dir',
            type=str,
            default="vocab_dict/learning_terminal_dict_1k_value.json",
            help='')
        parser.add_argument(
            '--type_dict_dir',
            type=str,
            default="vocab_dict/learning_terminal_dict_1k_type.json",
            help='')
        parser.add_argument('--vocab_size', type=int, default=1001, help='')
        """
        parser.add_argument('--train_data_dir',
                            type=str,
                            default="data/newctemp",
                            help='')
        parser.add_argument('--validate_data_dir',
                            type=str,
                            default="data/newctemp",
                            help='')
        """
        parser.add_argument('--train_data_dir',
                            type=str,
                            default="data/learning/train_data",
                            help='')
        parser.add_argument('--validate_data_dir',
                            type=str,
                            default="data/learning/validate_data",
                            help='')
        # args 获取命令行输入，和默认值。
        # 当需要读取参数时，在load中读取并设置。
        return parser
        """
        return {
            'optimizer': 'Adam',  # this key is not use.
            'learning_rate': 0.001,
            'graph_num_layers': 8,  # the layer num of ggnn.
            'max_epochs': 1500,
            'result_dir': 'trained_models',  # the dir of output.
            "dropout_rate":
            0.5,  # dropout_rate = 0, keep_prob = 1, dropout is disable.
            "do_model_load": False,
            "model_load_file":
            r"trained_models\model_save\CodeCompletion-Task1_Combine-ggnn_single_ast-None-Model_2021-03-10-11-50-44_21892_best_model.pt",
            "cur_epoch": 1,
            #--tasks--
            'max_nodes_per_batch':
            500,  # 50000
            "embedding_size":
            128,
            'graph_node_label_max_num_chars':
            19,  # 最大字符数。
            "device":
            "cuda",  # "cuda" or "cpu"
            "value_dict_dir": "dataProcessing/left_sequence_linux_terminal_dict_1k.json",
            "vocab_size": 1001,  # 0-999 is dict, 1000 is empty.
            "vocab_empty_idx": 1000,  # 0-999 is dict, 1000 is empty.

        }
        """

    @staticmethod
    def name() -> str:
        return "DGAP-Model"

    def __init__(self, args):
        self.args = args
        self.run_id = "_".join(
            [self.name(),
             time.strftime("%Y-%m-%d-%H-%M-%S"),
             str(getpid())])
        self._loaded_datasets = {}
        self.load_data()
        self.__make_model()

    @property
    def log_file(self):
        return osp.join(self.args.result_dir, "%s.log" % self.run_id)

    def log_line(self, msg):
        with open(self.log_file, 'a') as log_fh:
            log_fh.write(msg + '\n')
        print(msg)

    @property
    def best_model_file(self):
        return osp.join(
            self.args.result_dir,
            osp.join("model_save", "%s_best_model.pt" % self.run_id))

    def save_model(self, path: str) -> None:
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "params": vars(self.args),
        }
        torch.save(save_dict, path)

    def load_model(self, path) -> None:
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.args.cur_epoch = checkpoint['params']['cur_epoch']
        #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def __make_model(self) -> None:
        # 构造模型。
        modelCls, appendix_args = name_to_model_class(self.args.backbone_model, self.args)
        self.log_line("model_args:"+json.dumps(appendix_args))
        num_edge_types = self.make_task_input(None, get_num_edge_types=True)
        self.model = modelCls(
                        num_edge_types=num_edge_types,
                        in_features=self.args.graph_node_max_num_chars,
                        out_features=self.args.out_features,
                        embedding_out_features=self.args.h_features,
                        classifier_features=self.args.vocab_size,
                        embedding_num_classes=70,
                        dropout=self.args.dropout_rate,
                        device=self.args.device, output_model=self.args.output_model, **appendix_args)
        #self.model = GraphConvolution(self.params['embedding_size'], self.params['embedding_size'], is_sparse_adjacency=False)

        self.model.to(self.args.device)
        #self.model.apply(self.apply_weight_init)
        if self.args.load_model_file is not None:
            self.load_model(self.args.load_model_file)

        self.__make_train_step()

    def __make_train_step(self):
        # 设置优化器等参数。
        # use optimizer
        lr = self.args.lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        batchNumsPerEpoch = len(self._loaded_datasets[DataFold.TRAIN]) // self.args.batch_size + 1
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.args.lr_deduce_per_epoch*batchNumsPerEpoch, gamma=0.6, last_epoch=-1)
        # 模型所有参数统计
        self.args.num_model_params = sum(
            p.numel() for p in list(self.model.parameters()))  # numel()
        '''
        self.optimizer = torch.optim.Adam([{
            "params": value.parameters()
        } for key, value in self.all_model.items()],
                                          lr=lr)
        '''

    def load_data(self) -> None:
        """导入数据。数据会存储在self._loaded_data中。

        Raises:
            Exception: [description]
        """
        train_path = self.args.train_data_dir
        validate_path = self.args.validate_data_dir

        # if dataset dir not exists
        if not osp.exists(train_path) or not osp.exists(validate_path):
            raise Exception("train data or validate data dir not exists error")

        if not osp.exists(self.args.value_dict_dir) or not osp.exists(
                self.args.type_dict_dir):
            raise Exception("vocab dict not exists error")

        with open(self.args.value_dict_dir, 'r') as f:
            value_dict = json.load(f)
        with open(self.args.type_dict_dir, 'r') as f:
            type_dict = json.load(f)

        datasetCls, appendix_args, self.make_task_input = name_to_dataset_class(self.args.dataset_name, self.args)

        self._loaded_datasets[DataFold.TRAIN] = datasetCls(
            train_path,
            value_dict,
            type_dict,
            graph_node_max_num_chars=self.args.graph_node_max_num_chars,
            max_graph=20000, max_variable_candidates=self.args.max_variable_candidates, **appendix_args)
        self._loaded_datasets[DataFold.VALIDATION] = datasetCls(
            validate_path,
            value_dict,
            type_dict,
            self.args.graph_node_max_num_chars,
            max_graph=10000, **appendix_args)

    def criterion(self, y_score, y_true, criterion=torch.nn.CrossEntropyLoss()):
        loss = criterion(y_score, y_true)
        metrics = cal_metrics(F.softmax(y_score, dim=-1), y_true)
        return loss, metrics

    def __run_epoch(
        self,
        epoch_name: str,
        data: Iterable[Any],
        data_fold: DataFold,
        batch_size: int,
        quiet: Optional[bool] = False,
    ) -> Tuple[float]:
        """具体的每一轮训练。

        Args:
            epoch_name (str): 每一轮名称
            data (Iterable[Any]): 该轮的数据
            data_fold (DataFold): 是test还是train
            quiet (Optional[bool], optional): 当为真时，不显示任何信息。. Defaults to False.

        Returns:
            Tuple[float]: [description]
        """
        batch_iterator = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True if data_fold == DataFold.TRAIN else False,
            #num_workers=int(cpu_count()/2))
            num_workers=0)

        start_time = time.time()
        processed_graphs, processed_nodes, processed_batch = 0, 0, 0
        epoch_loss = 0.0
        task_metric_results = []

        for batch_data in batch_iterator:
            batch_data = batch_data.to(self.args.device)

            processed_graphs += batch_data.num_graphs
            processed_nodes += batch_data.num_nodes
            processed_batch += 1

            #self.torch_summary(batch_data)
            if data_fold == DataFold.TRAIN:
                '''with SummaryWriter(comment="model_test") as w:
                    w.add_graph(self.model, input_to_model=(
                        batch_data.feed_dict['input_adjacency_list'],
                        batch_data.feed_dict['unique_label_chars_one_hot'],
                        batch_data.feed_dict['input_node_labels_to_unique_labels'], 
                        batch_data.feed_dict['input_local_slot_node_idx'],
                        batch_data.feed_dict['input_local_candidate_node_ids'],))'''
                self.optimizer.zero_grad()
                self.model.train()
                '''
                logits = self.model(batch_data.x, batch_data.x_type,
                                    [batch_data.edge_index,
                                    #batch_data.edge_index_two_hop,
                                    #batch_data.edge_index_three_hop,
                                    #batch_data.edge_index_five_hop,
                                    batch_data.edge_index_ncs,
                                    #batch_data.edge_index_dfg,
                                    batch_data.edge_index_last_write,
                                    batch_data.edge_index_last_use,
                                    batch_data.edge_index_self_loop ],
                                    batch_data.batch, batch_data.num_graphs,
                                    batch_data.right_most,
                                    batch_data.candidate_id,
                '''
                task_batch_data = self.make_task_input(batch_data)
                logits = self.model(**task_batch_data)
                #logits = self.task.make_task_output(batch_data, output) 计算logits。
                loss, metrics = self.criterion(logits,
                                               batch_data.label)
                epoch_loss += loss.item()
                task_metric_results.append(metrics)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            if data_fold == DataFold.VALIDATION:
                self.model.eval()
                with torch.no_grad():
                    '''
                    logits = self.model(batch_data.x, batch_data.x_type,
                                        [batch_data.edge_index,
                                        #batch_data.edge_index_two_hop,
                                        #batch_data.edge_index_three_hop,
                                        #batch_data.edge_index_five_hop,
                                        batch_data.edge_index_ncs,
                                        #batch_data.edge_index_dfg,
                                        batch_data.edge_index_last_write,
                                        batch_data.edge_index_last_use,
                                        batch_data.edge_index_self_loop ],
                                        batch_data.batch, batch_data.num_graphs,
                                        batch_data.right_most,
                                        batch_data.candidate_id,
                    '''
                    """
                    logits = self.model(batch_data.x, [
                        batch_data.Child_index,
                        batch_data.NextToken_index,
                        batch_data.LastUse_index,
                        batch_data.LastWrite_index,
                        batch_data.LastLexicalUse_index,
                        batch_data.ComputedFrom_index,
                        batch_data.GuardedByNegation_index,
                        batch_data.GuardedBy_index,
                        batch_data.FormalArgName_index,
                        batch_data.ReturnsTo_index,
                        batch_data.SelfLoop_index,
                    ], batch_data.slot_id, batch_data.candidate_ids,
                                        batch_data.candidate_masks)
                    """
                    task_batch_data = self.make_task_input(batch_data)
                    logits = self.model(**task_batch_data)
                    loss, metrics = self.criterion(logits,
                                                   batch_data.label)
                    epoch_loss += loss.item()
                    task_metric_results.append(metrics)

            if not quiet:
                print("Runing %s, batch %i (has %i graphs). Loss so far: %.4f" %
                      (epoch_name, processed_batch, batch_data.num_graphs,
                       epoch_loss / processed_batch),
                      end="\r")

        epoch_time = time.time() - start_time
        per_graph_loss = epoch_loss / processed_batch
        graphs_per_sec = processed_graphs / epoch_time
        nodes_per_sec = processed_nodes / epoch_time

        return per_graph_loss, task_metric_results, processed_graphs, processed_batch, graphs_per_sec, nodes_per_sec, processed_graphs, processed_nodes

    def train(self, quiet=False):
        """训练函数。调用train_epoch训练每一个epoch，获取输出后进行输出。

        Args:
            quiet (bool, optional): [description]. Defaults to False.
        """
        self.log_line(json.dumps(vars(self.args), indent=4))
        total_time_start = time.time()

        (best_valid_metric, best_val_metric_epoch,
         best_val_metric_descr) = (float("+inf"), 0, "")
        for epoch in range(self.args.cur_epoch, self.args.max_epochs + 1):
            self.log_line("== Epoch %i" % epoch)
            # --train
            train_loss, train_task_metrics, train_num_graphs, train_num_batchs, train_graphs_p_s, train_nodes_p_s, train_graphs, train_nodes = self.__run_epoch(
                "epoch %i (training)" % epoch,
                self._loaded_datasets[DataFold.TRAIN],
                DataFold.TRAIN,
                self.args.batch_size,
                quiet=quiet)

            if not quiet:
                print("\r\x1b[K", end='')  #该函数意义将光标回到该行开头，并擦除整行。
            self.log_line(
                " Train: loss: %.5f || %s || graphs/sec: %.2f | nodes/sec: %.0f | graphs: %.0f | nodes: %.0f | lr: %0.6f"
                %
                (train_loss,
                 pretty_print_epoch_task_metrics(
                     train_task_metrics, train_num_graphs, train_num_batchs),
                 train_graphs_p_s, train_nodes_p_s, train_graphs, train_nodes, self.scheduler.get_last_lr()[0]))

            # --validate
            valid_loss, valid_task_metrics, valid_num_graphs, valid_num_batchs, valid_graphs_p_s, valid_nodes_p_s, test_graphs, test_nodes = self.__run_epoch(
                "epoch %i (validation)" % epoch,
                self._loaded_datasets[DataFold.VALIDATION],
                DataFold.VALIDATION,
                self.args.batch_size,
                quiet=quiet)

            early_stopping_metric = cal_early_stopping_metric(
                valid_task_metrics)
            valid_metric_descr = pretty_print_epoch_task_metrics(
                valid_task_metrics, valid_num_graphs, valid_num_batchs)
            if not quiet:
                print("\r\x1b[K", end='')
            self.log_line(
                " valid: loss: %.5f || %s || graphs/sec: %.2f | nodes/sec: %.0f | graphs: %.0f | nodes: %.0f | lr: %0.6f"
                % (valid_loss, valid_metric_descr, valid_graphs_p_s,
                   valid_nodes_p_s, test_graphs, test_nodes, self.scheduler.get_last_lr()[0]))

            if early_stopping_metric < best_valid_metric:
                self.args.cur_epoch = epoch + 1
                self.save_model(self.best_model_file)
                self.log_line(
                    "  (Best epoch so far, target metric decreased to %.5f from %.5f. Saving to '%s')"
                    % (early_stopping_metric, best_valid_metric,
                       self.best_model_file))
                best_valid_metric = early_stopping_metric
                best_val_metric_epoch = epoch
                best_val_metric_descr = valid_metric_descr

    def test(self):
        pass


if __name__ == "__main__":
    cc_cls = CodeCompletion
    parser = cc_cls.default_args()
    args = parser.parse_args([
        "--lr=0.7",
    ])
    cc = cc_cls(args)
    print(json.dumps(vars(args), indent=4))

    from dataProcessing import GraphData
    from torch_geometric.data import DataLoader
    from torch_geometric.utils import to_undirected, add_self_loops
    import torch
    D = 16
    V = 5
    edge = torch.tensor([[0, 1], [1, 2], [2, 4], [3, 4]])
    edge = to_undirected(edge.t())
    edge, weight = add_self_loops(edge)
    adj1 = torch.sparse_coo_tensor(edge, torch.ones(edge.shape[1]), (V, V))
    adj2 = torch.sparse.mm(adj1, adj1)
    adj3 = torch.sparse.mm(adj2, adj1)
    adj4 = torch.sparse.mm(adj3, adj1)
    adj5 = torch.sparse.mm(adj4, adj1)

    adj1 = adj1.coalesce()
    adj2 = adj2.coalesce()
    adj3 = adj3.coalesce()
    adj4 = adj4.coalesce()
    adj5 = adj5.coalesce()
    # 获取1， 3， 5阶邻居只能通过遍历来。通过矩阵乘法不靠谱啊。
    # 这里先弄一个数据出来。

    adj1_index = adj1.indices()
    adj3_index = adj3.indices()
    adj5_index = adj5.indices()
    node_init_str = torch.randint(low=0, high=70, size=(V, D))
    data = GraphData(adj1_index, adj3_index, adj5_index, 1, node_init_str)

    dataList = [data for _ in range(10)]
    dataloader = DataLoader(dataList, batch_size=5, shuffle=True)
    gam = DGAP(D, 2 * D, D, 70, 8)
    for batch_data in dataloader:
        out = gam(batch_data.x, batch_data.edge_index)
        print(out)

    print("")
