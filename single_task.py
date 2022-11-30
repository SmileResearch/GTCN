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

from utils import name_to_dataloader, name_to_model, name_to_output_model, DataFold
from utils import pretty_print_epoch_task_metrics, cal_early_stopping_metric, cal_metrics
from utils import set_seed

import warnings
warnings.filterwarnings('ignore')

from utils.model_metrics import roc_curve_point


class Single_Task:
    @classmethod
    def default_args(cls):
        parser = argparse.ArgumentParser(description="Single Task Model Detection")

        parser.add_argument('--output_model', type=str, default="vm", help='the type of downstream task, Optional[vm, cc]')
        parser.add_argument('--max_variable_candidates', type=int, default=10, help='the max candidates num in vm task.')
        parser.add_argument('--result_dir', type=str, default="trained_models/", help='the directory of saving result.')
        parser.add_argument('--load_model_file', type=str, default=None, help='when is not None, the model will load checkpoint first, and the load_model_file is the checkpoint path.')
        parser.add_argument('--backbone_model',
                            type=str,
                            default="tensor_gcn",
                            help='the backbone model.')
        parser.add_argument('--optimizer', type=str, default="Adam", help='the optimizer.')
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--max_epochs', type=int, default=1500, help='max epochs')
        parser.add_argument('--cur_epoch', type=int, default=1, help='use for load model. Can\'t be specified.')
        parser.add_argument('--batch_size', type=int, default=128, help='')
        parser.add_argument('--dropout_rate', type=float, default=0., help='keep_prob = 1-dropout_rate')
        parser.add_argument('--h_features', type=int, default=128, help='the hidden layer size of backbone model')
        parser.add_argument('--out_features', type=int, default=128, help='the output layer size of backbone model')
        parser.add_argument('--graph_node_max_num_chars', type=int, default=19, help='the max character nums of a variable.')
        parser.add_argument('--max_node_per_graph', type=int, default=50, help='the max node per graph')
        parser.add_argument('--device', type=str, default="cuda", help='')


        # 数据集相关
        parser.add_argument('--slice_edge_type', type=str, default="[0,1,2,3,4,5,6,7,8]", help='the chose graphs for model.')
        parser.add_argument('--num_edge_types', type=int, default=3, help='num of edges\' type, equals to length of slice_edge_type')
        parser.add_argument('--train_data_dir',
                            type=str,
                            default="data/lrtemp/train_data",
                            help='train data dir.')
        parser.add_argument('--validate_data_dir',
                            type=str,
                            default="data/csharp/validate_data",
                            help='validate data dir')
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
        parser.add_argument('--dataset_num_workers', type=int, default=0, help='the cpu nums for load data.')
        
        parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
        parser.add_argument("--slot_singal", default="<SLOT>", type=str,
                        help="slot singal during training, default=<SLOT>")
        parser.add_argument('--max_graph', type=int, default=100000,
                        help="max data graph.")
        
        parser.add_argument('--notes', type=str, default="None", help=' notes will show in result file title.')
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
        # load data according to train_data_dir, validate_data_dir
        train_path = self.args.train_data_dir
        validate_path = self.args.validate_data_dir


        if not osp.exists(train_path) or not osp.exists(validate_path):
            raise Exception("train data or validate data dir not exists error!")

        if not osp.exists(self.args.value_dict_dir) or not osp.exists(
                self.args.type_dict_dir):
            raise Exception("vocab dict not exists error!")


        # init dataset
        self._loaded_datasets = dict()
        
        # load data
        if self.args.dataset_num_workers is None:
            self.args.dataset_num_workers = int(cpu_count() / 2)
        self._loaded_datasets[DataFold.TRAIN] = name_to_dataloader(self.args.dataset_name, train_path, DataFold.TRAIN, self.args, num_workers=self.args.dataset_num_workers)
        self._loaded_datasets[DataFold.VALIDATION] = name_to_dataloader(self.args.dataset_name, validate_path, DataFold.VALIDATION, self.args, num_workers=self.args.dataset_num_workers)

        self.args.train_data_nums = self._loaded_datasets[DataFold.TRAIN].data_nums
        self.args.valid_data_nums = self._loaded_datasets[DataFold.VALIDATION].data_nums

    def save_model(self, path, others_dict=None):
        # save model to path.
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
        # load model according to path
        if not os.path.exists(path):
            raise Exception("paths not exists %s" % path)
        self.model = joblib.load(os.path.join(path, "model.joblib"))
        self.output_model = joblib.load(os.path.join(path, "output_model.joblib"))
        self.optimizer = torch.load(os.path.join(path, "optimizer.pt"))
        with open(os.path.join(path, "params.json"), "r") as f:
            args = json.load(f)
            
        self.args.cur_epoch = args['cur_epoch']
    
    def __make_model(self) -> None:
        # build the model according to the arguments.
        if self.args.load_model_file is not None:
            self.log_line(f"load model:{self.args.load_model_file}")
            self.load_model(self.args.load_model_file)
            
            assert self.model.device == self.args.device, "the device of loaded model conflict with args.device"
            
        else:
            self.model = name_to_model(self.args.backbone_model, self.args)
            self.output_model = name_to_output_model(self.args.output_model, self.args)
            
            self.model.to(self.args.device)
            self.output_model.to(self.args.device)
            
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

            # caculate the num of args
            self.args.num_model_params = sum(
                p.numel() for p in list(self.model.parameters()))  # numel()
    
    
    def __run_epoch(
        self,
        epoch_name: str,
        data: Iterable[Any],
        data_fold: DataFold,
        quiet: Optional[bool] = False,
    ) -> Tuple[float]:
        """train one epoch

        Args:
            epoch_name (str):
            data (Iterable[Any]): the data of cur epoch
            data_fold (DataFold): Optional[Train, Valid]
            quiet (Optional[bool], optional): Whether to show output. Defaults to False.

        Returns:
            Tuple[float]: [Return the tuple of metrics.]
        """
        
        start_time = time.time()
        processed_graphs, processed_nodes, processed_batch = 0, 0, 0
        epoch_loss = 0.0
        task_metric_results = []

        for batch_data in data.batch_generator():

            processed_graphs += batch_data["num_graphs"]
            processed_nodes += batch_data["num_nodes"]
            processed_batch += 1


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
                    self.scheduler.step()

            if data_fold == DataFold.VALIDATION:
                self.model.eval() 
                self.output_model.eval()
                with torch.no_grad():

                    output = self.model(**batch_data)
                    logits, loss, metrics = self.output_model(output, **batch_data)
                    epoch_loss += loss.item()
                    task_metric_results.append(metrics)

            if not quiet:
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
        
        self.log_line(json.dumps(vars(self.args), indent=4))

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
                print("\r\x1b[K", end='') 
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
                # set early stopping
                self.args.cur_epoch = epoch + 1
                self.save_model(self.best_model_file)
                self.log_line(
                    "  (Best epoch so far, target metric decreased to %.5f from %.5f. Saving to '%s')"
                    % (early_stopping_metric, best_valid_metric,
                       self.best_model_file))
                best_valid_metric = early_stopping_metric
                
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