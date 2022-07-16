import argparse
from enum import Enum
import time
import os.path as osp
from os import getpid
import json
from typing import Any, Dict, Optional, Tuple, List, Iterable, Union
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
from random import shuffle
from multiprocessing import cpu_count

import joblib
import os

# 导入本地库
from utils import name_to_dataloader, name_to_model, name_to_output_model, DataFold
from utils import pretty_print_epoch_task_metrics, cal_early_stopping_metric, cal_metrics
from utils import set_seed

import warnings
warnings.filterwarnings('ignore')

from utils.model_metrics import roc_curve_point

# 重构当前代码

class Single_Task:
    @classmethod
    def default_args(cls):
        parser = argparse.ArgumentParser(description="Single Task 模型预测")

        
        # 任务相关
        parser.add_argument('--output_model', type=str, default="vm", help='输出层的任务，可选varmisuse, codecompletion')
        parser.add_argument('--max_variable_candidates', type=int, default=10, help='')
        
        # 输出相关
        parser.add_argument('--result_dir', type=str, default="trained_models/", help='')

        # 读取模型、继续训练
        parser.add_argument('--load_model_file', type=str, default=None, help='当这个值为False时，不导入模型。当部位None时，则是模型存储的地址。')
        #parser.add_argument('--load_model_file', type=str, default="trained_models/model_save/Single-Task_2022-05-07-18-27-35_26536_resgagn_best_model", help='')

        # 模型训练
        parser.add_argument('--backbone_model',
                            type=str,
                            default="tensor_gcn",
                            help='the backbone model of features extract')
        parser.add_argument('--optimizer', type=str, default="Adam", help='TODO:暂时不接受选择')
        parser.add_argument('--lr', type=float, default=0.001, help='')
        parser.add_argument('--lr_deduce_per_epoch', type=int, default=10, help='')
        parser.add_argument('--max_epochs', type=int, default=1500, help='')
        parser.add_argument('--cur_epoch', type=int, default=1, help='用做读取checkpoint再训练的参数，手动设置无效。')
        parser.add_argument('--batch_size', type=int, default=128, help='')
        parser.add_argument('--dropout_rate', type=float, default=0., help='keep_prob = 1-dropout_rate')
        parser.add_argument('--h_features', type=int, default=128, help='')
        parser.add_argument('--out_features', type=int, default=128, help='')
        parser.add_argument('--graph_node_max_num_chars', type=int, default=19, help='图中节点的初始特征维度')
        parser.add_argument('--max_node_per_graph', type=int, default=50, help='一个图最多的节点个数')
        parser.add_argument('--device', type=str, default="cuda", help='')


        # 数据集相关
        parser.add_argument('--slice_edge_type', type=str, default="[0,1,2,3,4,5,6,7,8]", help='数据集中边的数量。')
        parser.add_argument('--num_edge_types', type=int, default=3, help='数据集中边的数量。')
        parser.add_argument('--train_data_dir',
                            type=str,
                            default="data/lrtemp/train_data",
                            help='')
        parser.add_argument('--validate_data_dir',
                            type=str,
                            default="data/csharp/validate_data",
                            help='')
        parser.add_argument('--dataset_name', type=str, default="csharp", help='the name of the dataset. optional:[python, csharp]')
        parser.add_argument(
            '--value_dict_dir',
            type=str,
            default="vocab_dict/python_terminal_dict_1k_value.json",
            help='')
        parser.add_argument(
            '--type_dict_dir',
            type=str,
            default="vocab_dict/python_terminal_dict_1k_type.json",
            help='')
        parser.add_argument('--dataset_num_workers', type=int, default=0, help='如果设置为None，则启用cpu_count/2个进程。若为0，则不预先加载。这是个坑，每个batch都会重新创建进程，竟然不是进程池，很难以理解。设置为0，不用想，创建进程的开销比读取数据的开销还大。')
        
        # 其他参数设置
        parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
        parser.add_argument("--slot_singal", default="<SLOT>", type=str,
                        help="slot singal during training, default=<SLOT>")
        parser.add_argument('--max_graph', type=int, default=100000,
                        help="max data graph. max_graph = None时， 为使用所有数据")
        
        parser.add_argument('--notes', type=str, default="None", help=' notes ')
        return parser

    @staticmethod
    def name() -> str:
        return "Single_Task"

    def __init__(self, args):
        self.args = args
        # pre parse args
        if self.args.slice_edge_type is not None:
            self.args.slice_edge_type = json.loads(self.args.slice_edge_type)
            self.args.num_edge_types=len(self.args.slice_edge_type)
        if self.args.max_graph is None:
            import sys
            self.args.max_graph = sys.maxsize
        
        set_seed(args)
        self.run_id = "_".join([self.name(), time.strftime("%Y_%m_%d_%H_%M_%S"), str(getpid()), self.args.backbone_model, self.args.notes])
        self.__load_data()
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
        return osp.join(self.args.result_dir, osp.join("model_save", "%s" % (self.run_id)))
    
    
    def __load_data(self):
        train_path = self.args.train_data_dir
        validate_path = self.args.validate_data_dir

        # 数据地址或者字典地址有误则报错。
        if not osp.exists(train_path) or not osp.exists(validate_path):
            raise Exception("train data or validate data dir not exists error!")

        if not osp.exists(self.args.value_dict_dir) or not osp.exists(
                self.args.type_dict_dir):
            # TODO: value_dict, type_dict待改。 
            raise Exception("vocab dict not exists error!")


        # 数据初始化
        self._loaded_datasets = dict()
        
        # 导入数据
        if self.args.dataset_num_workers is None:
            self.args.dataset_num_workers = int(cpu_count() / 2)
        self._loaded_datasets[DataFold.TRAIN] = name_to_dataloader(self.args.dataset_name, train_path, DataFold.TRAIN, self.args, num_workers=self.args.dataset_num_workers)
        self._loaded_datasets[DataFold.VALIDATION] = name_to_dataloader(self.args.dataset_name, validate_path, DataFold.VALIDATION, self.args, num_workers=self.args.dataset_num_workers)

        self.args.train_data_nums = self._loaded_datasets[DataFold.TRAIN].data_nums
        self.args.valid_data_nums = self._loaded_datasets[DataFold.VALIDATION].data_nums

    def save_model(self, path, others_dict=None):
        if not os.path.exists(path):
            os.makedirs(path)
        joblib.dump(self.model, os.path.join(path, "model.joblib"))
        joblib.dump(self.output_model, os.path.join(path, "output_model.joblib"))
        torch.save(self.optimizer,os.path.join(path, "optimizer.pt"))
        with open(os.path.join(path, "params.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)
        if others_dict is not None:
            for key in others_dict:
                joblib.dump(others_dict[key], os.path.join(path, key+".joblib"))
    
    def load_model(self, path):
        if not os.path.exists(path):
            raise Exception("文件夹不存在 %s" % path)
        self.model = joblib.load(os.path.join(path, "model.joblib"))
        self.output_model = joblib.load(os.path.join(path, "output_model.joblib"))
        self.optimizer = torch.load(os.path.join(path, "optimizer.pt"))
        with open(os.path.join(path, "params.json"), "r") as f:
            args = json.load(f)
            
        self.args.cur_epoch = args['cur_epoch']
    
    def __make_model(self) -> None:
        # 构造模型
        # 影响模型的三个方面：选定的模型、输出层、是否导入模型（模型保存二次训练）
        # 首先判断是否导入模型，是的话导入，否的话创建模型
        
        if self.args.load_model_file is not None:
            self.log_line(f"导入模型:{self.args.load_model_file}")
            self.load_model(self.args.load_model_file)
            
            assert self.model.device == self.args.device, "导入模型的device和设置不一致"
            
        else:
            # 构造模型
            self.model = name_to_model(self.args.backbone_model, self.args)
            self.output_model = name_to_output_model(self.args.output_model, self.args)
            
            self.model.to(self.args.device)
            self.output_model.to(self.args.device)
            
            # 构造优化器
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

            # 根据运行动态改边学习率。先不改变。
            # batchNumsPerEpoch = len(self._loaded_datasets[DataFold.TRAIN]) // self.args.batch_size + 1
            # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.args.lr_deduce_per_epoch*batchNumsPerEpoch, gamma=0.6, last_epoch=-1)
            
            # 模型所有参数统计
            self.args.num_model_params = sum(
                p.numel() for p in list(self.model.parameters()))  # numel()
    
    
    def __run_epoch(
        self,
        epoch_name: str,
        data: Iterable[Any],
        data_fold: DataFold,
        quiet: Optional[bool] = False,
    ) -> Tuple[float]:
        """具体的每一轮训练。

        Args:
            epoch_name (str): 每一轮名称
            data (Iterable[Any]): 该轮的数据
            data_fold (DataFold): 是test还是train
            quiet (Optional[bool], optional): 当为真时，不显示任何信息。. Defaults to False.

        Returns:
            Tuple[float]: [返回一些metrics]
        """
        
        # 记录部分参数
        # processed_* 每一个epoch信息的记录。
        start_time = time.time()
        processed_graphs, processed_nodes, processed_batch = 0, 0, 0
        epoch_loss = 0.0
        task_metric_results = []

        for batch_data in data.batch_generator():

            processed_graphs += batch_data["num_graphs"]
            processed_nodes += batch_data["num_nodes"]
            processed_batch += 1


            # TODO: 模型输出后，再根据输出层产生输出。
            if data_fold == DataFold.TRAIN:
                self.optimizer.zero_grad()
                self.model.train()
                self.output_model.train()
                output = self.model(**batch_data)
                logits, loss, metrics = self.output_model(output, **batch_data)
                epoch_loss += loss.item()
                task_metric_results.append(metrics)

                loss.backward()
                self.optimizer.step()
                if hasattr(self, "scheduler"):
                    # 如果设置了学习率的动态变化的化，对scheduler进行更新。
                    self.scheduler.step()

            if data_fold == DataFold.VALIDATION:
                self.model.eval() # eval和no_grad设置一个就可。这里为了保险都设置了。TODO: 看看这能不能修改。
                self.output_model.eval()
                with torch.no_grad():

                    output = self.model(**batch_data)
                    logits, loss, metrics = self.output_model(output, **batch_data)
                    epoch_loss += loss.item()
                    task_metric_results.append(metrics)

            if not quiet:
                # end="\r"意思是，打印后，光标回到句子头部。下次打印的时候覆盖这一行。
                print("Runing %s, batch %i (has %i graphs). Loss so far: %.4f" %
                      (epoch_name, processed_batch, batch_data["num_graphs"],
                       epoch_loss / processed_batch),
                      end="\r")
                

        epoch_time = time.time() - start_time
        per_graph_loss = epoch_loss / processed_batch
        graphs_per_sec = processed_graphs / epoch_time
        nodes_per_sec = processed_nodes / epoch_time

        return per_graph_loss, task_metric_results, processed_graphs, processed_batch, graphs_per_sec, nodes_per_sec, processed_graphs, processed_nodes

    
    def train(self, quiet=False):
        """对模型进行训练

        Args:
            quiet (bool, optional): _description_. Defaults to False.
        """
        
        # 在日志中打印当前的设置参数。
        self.log_line(json.dumps(vars(self.args), indent=4))

        # 存储当前训练过程中的最好参数
        (best_valid_metric, best_val_metric_epoch,
         best_val_metric_descr) = (float("+inf"), 0, "")
        for epoch in range(self.args.cur_epoch, self.args.max_epochs + 1):
            self.log_line("== Epoch %i" % epoch)
            # --train
            train_loss, train_task_metrics, train_num_graphs, train_num_batchs, train_graphs_p_s, train_nodes_p_s, train_graphs, train_nodes = self.__run_epoch(
                "epoch %i (training)" % epoch,
                self._loaded_datasets[DataFold.TRAIN],
                DataFold.TRAIN,
                quiet=quiet)

            if not quiet:
                print("\r\x1b[K", end='')  #该函数意义将光标回到该行开头，并擦除整行。
            self.log_line(
                " Train: loss: %.5f || %s || graphs/sec: %.2f | nodes/sec: %.0f | graphs: %.0f | nodes: %.0f | lr: %0.6f"
                %
                (train_loss,
                 pretty_print_epoch_task_metrics(
                     train_task_metrics, train_num_graphs, train_num_batchs),
                 train_graphs_p_s, train_nodes_p_s, train_graphs, train_nodes, self.optimizer.state_dict()["param_groups"][0]['lr'])) #, self.scheduler.get_last_lr()[0]

            # --validate
            valid_loss, valid_task_metrics, valid_num_graphs, valid_num_batchs, valid_graphs_p_s, valid_nodes_p_s, test_graphs, test_nodes = self.__run_epoch(
                "epoch %i (validation)" % epoch,
                self._loaded_datasets[DataFold.VALIDATION],
                DataFold.VALIDATION,
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
                   valid_nodes_p_s, test_graphs, test_nodes, self.optimizer.state_dict()["param_groups"][0]['lr'])) # , self.scheduler.get_last_lr()[0]

            if early_stopping_metric < best_valid_metric:
                # 设置早停。
                self.args.cur_epoch = epoch + 1
                self.save_model(self.best_model_file)
                self.log_line(
                    "  (Best epoch so far, target metric decreased to %.5f from %.5f. Saving to '%s')"
                    % (early_stopping_metric, best_valid_metric,
                       self.best_model_file))
                best_valid_metric = early_stopping_metric
                # 保存每一个batch的fpr,tpr，然后将其保存下来，最后读取再绘制。[{"fpr":[], "tpr":[], "auc":[]}].存储为json。
                
                best_val_metric_epoch = epoch
                best_val_metric_descr = valid_metric_descr

    

if __name__ == '__main__':
    
    
    cc_cls = Single_Task
    
    parser = cc_cls.default_args()
    args = parser.parse_args()
    cc = cc_cls(args)
    print("Training Start, luck bless me, plz!")
    cc.train()
    print("Training Done!")