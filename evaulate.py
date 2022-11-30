import argparse

import logging
import time
import json
from os import getpid
import os
from multiprocessing import cpu_count
import joblib
import torch
import torch.nn.functional as F
import numbers
from itertools import compress
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm, trange

# 自己的模块
from utils import name_to_dataloader, name_to_model, name_to_output_model, DataFold
from utils import roc_curve_point


class Evaulate:
    # 导入模型，对数据进行评估。
    # 选择模式，根据参数确定模式进行评估，
    # roc
    # case study
    
    @classmethod
    def default_args(cls):
        parser = argparse.ArgumentParser(description="Single Task 模型预测", add_help=False)

        # load model
        parser.add_argument('--load_model_file', type=str, default="trained_models/model_save/Single-Task_2022-05-28-11-18-56_17866_tensor_gcn_tensor_gcn_best_model", help='the checkpoint path of choose model.')
        
        # set device
        parser.add_argument('--device', type=str, default="cuda", help='')
        
        # load data
        parser.add_argument('--evaulate_data_dir',
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
        
        
        parser.add_argument('--result_dir', type=str, default="trained_models/", help='')
        parser.add_argument('--notes', type=str, default="None", help=' notes ')
        
        def add_bool_arg(parser, name, help="", default=False):
            group = parser.add_mutually_exclusive_group(required=False)
            group.add_argument('--' + name, dest=name, action='store_true', help=help)
            group.add_argument('--no-' + name, dest=name, action='store_false')
            parser.set_defaults(**{name:default})
        
        
        parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='evaluate the data. can set --roc to caculate roc. set --case to case analysis.')
        
        # 评估选项
        add_bool_arg(parser, "roc", help="do roc evaluate.", default=False) 
        add_bool_arg(parser, "case_analysis", help="output detail probability or each variable.", default=True) 
        add_bool_arg(parser, "only_different_case", help="following case analysis, when is True, only output the wrong case.", default=True) 
        
        
        return parser

    @staticmethod
    def name() -> str:
        return "Evaulate"

    def __init__(self, args):
        self.args = args
        # pre parse args
        self.args.run_id = "_".join([self.name(), time.strftime("%Y_%m_%d_%H_%M_%S"), str(getpid()), self.args.notes])
        
        self.evaulate_dir = os.path.join(self.args.result_dir, "evaulate")
        self.evaulate_log_dir = os.path.join(self.evaulate_dir, "log")
        self.evaulate_out_dir = os.path.join(self.evaulate_dir, "output", self.args.run_id)
        for temp_path in [self.evaulate_dir, self.evaulate_log_dir, self.evaulate_out_dir]:
            if not os.path.exists(temp_path):
                os.makedirs(temp_path)
        logging.basicConfig(filename=os.path.join(self.evaulate_log_dir, self.args.run_id+".log"), format='%(asctime)s - %(levelname)s - %(name)s :   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
        self.logger = logging.getLogger(self.args.run_id)
        
        self.__load_model_and_args(self.args.load_model_file)
        self.__load_data()

    def __load_data(self):
        self.logger.info("load dataset")
        evaulate_data_dir = self.args.evaulate_data_dir

        if not os.path.exists(evaulate_data_dir):
            raise Exception("train data or validate data dir not exists error!")

        if not os.path.exists(self.args.value_dict_dir) or not os.path.exists(
                self.args.type_dict_dir):
            raise Exception("vocab dict not exists error!")


        # dataset init
        self._loaded_datasets = dict()
        
        # load data
        if self.args.dataset_num_workers is None:
            self.args.dataset_num_workers = int(cpu_count() / 2)
            
        self._loaded_datasets[DataFold.TEST] = name_to_dataloader(self.args.dataset_name, evaulate_data_dir, DataFold.TEST, self.args, num_workers=self.args.dataset_num_workers)

        self.args.eavulate_data_nums = self._loaded_datasets[DataFold.TEST].data_nums
        self.logger.info("load dataset over!")

        
    
    def __load_model_and_args(self, path):
        self.logger.info("load model:{}".format(path))
        if not os.path.exists(path):
            raise Exception("path do not exists %s" % path)
        self.model = joblib.load(os.path.join(path, "model.joblib"))
        self.output_model = joblib.load(os.path.join(path, "output_model.joblib"))
        with open(os.path.join(path, "params.json"), "r") as f:
            args = json.load(f)
            for key, value in args.items():
                if key not in self.args:
                    if isinstance(value, numbers.Number):
                        exec(f"self.args.{key}={value}")
                    else:
                        exec(f"self.args.{key}='{value}'")
                        
        if self.args.slice_edge_type is not None:
            self.args.slice_edge_type = json.loads(self.args.slice_edge_type)
            self.args.num_edge_types=len(self.args.slice_edge_type)
        
        self.logger.info("load model done!")
    
    def output_result(self, json_object, filename):
        # output json object. all result save in json.
        path = os.path.join(self.evaulate_out_dir, filename+".json")
        with open(path, "w") as f:
            json.dump(json_object, f)
        self.logger.info("save results to:"+path)

    def roc_plot(self):
        # output roc points for plots.
        self.logger.info("start roc evaluate")
        logits_cat = []
        label_cat = []
        for batch_data in tqdm(self._loaded_datasets[DataFold.TEST].batch_generator(), total=self._loaded_datasets[DataFold.TEST].size):

            self.model.eval()
            self.output_model.eval()
            with torch.no_grad():

                output = self.model(**batch_data)
                logits, loss, labels, metrics = self.output_model(output, **batch_data, output_label=True)
                logits_cat.append(logits)
                label_cat.append(labels)
        
        logits_cat = torch.cat(logits_cat, dim=0)   
        label_cat = torch.cat(label_cat, dim=0)
        roc_dict = roc_curve_point(logits_cat, label_cat, classifier_nums=logits_cat.shape[-1])
        self.output_result(roc_dict, filename="roc")
        self.logger.info("roc evaluate done!")

    def case_analysis_vm(self):
        # case study for vm task.
        # json format:{"result":[{"origin":str, "predict":str, "score":score}], "slot_node_index":slot_node_index, "filename":filename}
        self.logger.info("进行样本详细评估")
        logits_cat = []
        labels_cat = []
        slot_node_name = []
        candidate_node_name = []
        slot_location = []
        file_name = []
        for batch_data in tqdm(self._loaded_datasets[DataFold.TEST].batch_generator(), total=self._loaded_datasets[DataFold.TEST].size):
            self.model.eval() 
            self.output_model.eval()
            with torch.no_grad():

                output = self.model(**batch_data)
                logits, loss, labels, metrics = self.output_model(output, **batch_data, output_label=True)
                logits_cat.append(logits)
                labels_cat.append(labels)
                slot_node_name.extend(batch_data["batch_data"].slot_node_name)
                candidate_node_name.extend(batch_data["batch_data"].candidate_node_name)
                slot_location.extend(batch_data["batch_data"].slot_location)
                file_name.extend(batch_data["batch_data"].file_name)
        logits_cat = torch.cat(logits_cat, dim=0)
        labels_cat = torch.cat(labels_cat, dim=0)
        logits_list = logits_cat.tolist()
        
        show_index = range(len(file_name))
        if self.args.only_different_case:
            # only show wrong predict case.
            predict= torch.argmax(logits_cat, dim=1)
            filter_index = (predict!=labels_cat).tolist() # List:[True, False, True ...] True means predict!=labels_cat
            show_index = compress(show_index, filter_index)

        output = []
        for i in show_index:
            cur_result_dict = dict()
            cur_result_dict["filename"] = file_name[i][0]
            cur_result_dict["slot_location"] = slot_location[i][0]
            
            cur_slot_node = slot_node_name[i][0]
            cur_candidate_node = candidate_node_name[i][0] 
            cur_logits = logits_list[i]
            cur_predict_list = []
            for index, candidate in enumerate(cur_candidate_node[:self.args.max_variable_candidates]):
                cur_predict_list.append({
                    "origin":cur_slot_node,
                    "predict":candidate,
                    "score":cur_logits[index],
                })
            
            cur_result_dict["result"] = cur_predict_list
            
            output.append(cur_result_dict)
        
        self.output_result(output, filename="case_analysis")
        self.logger.info("case analysis done!")
        
            
    

    def evaulate(self):
        if self.args.roc:
            self.roc_plot()
            
        if self.args.case_analysis:
            if self.args.output_model=="vm":
                self.case_analysis_vm()
            elif self.args.output_model == "cc":
                self.case_analysis_cc()





if __name__=="__main__":
    cc_cls = Evaulate
    
    parser = cc_cls.default_args()
    args = parser.parse_args()
    cc = cc_cls(args)
    print("Start Evaulate:")
    cc.evaulate()
    print("Evaulate Over!")