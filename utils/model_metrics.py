import torch
import numpy as np
from typing import Any, Dict, Optional, Tuple, List, Iterable
import copy
from sklearn.metrics import classification_report
from sklearn.metrics import top_k_accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
#TODO: 所有metrics待检查是否有误

def cal_metrics(y_score, y_true, classifier_nums):
    with torch.no_grad():
        metrics = {}
        #print(y_score)
        #print(y_true)
        #y_true_np = y_true.cpu().detach().numpy()

        # 测top-1, 到 top5, accuracy, f1-score, 
        logits = y_score.detach().cpu().numpy()
        y_trues = y_true.detach().cpu().numpy()
        y_preds = np.argmax(logits, axis=1)
        top_dict = {f"top{i}": top_k_accuracy_score(y_trues, logits, k=i, normalize=True, labels=list(range(classifier_nums))) for i in range(1, classifier_nums+1)}
        # #precision, recall, f1_score, support = precision_recall_fscore_support(y_trues, y_preds, average='micro', labels=list(range(classifier_nums)))
        # precision, recall, f1_score, support =precision_recall_fscore_support(y_trues, y_preds, average='micro', labels=list(range(classifier_nums)) )
        # print(precision, recall, f1_score)
        # #top5 = top_5(y_score, y_true)
        # #metrics.update(top5)
        # roc_score = roc_auc_score(y_trues, logits, multi_class="ovr", average="macro", labels=list(range(classifier_nums)) )
        # acc = accuracy_score(y_trues, y_preds)
        
        # recall值用logits，多分类计算
        # 其他值用二分类计算。
        y_trues_binary = label_binarize(y_trues, classes=range(classifier_nums))
        y_preds_binary = label_binarize(y_preds, classes=range(classifier_nums))
        y_trues_binary=np.reshape(y_trues_binary, (-1))
        y_preds_binary=np.reshape(y_preds_binary, (-1))
        logits_binary = np.reshape(logits, (-1))
        precision, recall, f1_score, support =precision_recall_fscore_support(y_trues_binary, y_preds_binary, average="binary" )
        roc_score = roc_auc_score(y_trues_binary, logits_binary)
        acc = accuracy_score(y_trues_binary, y_preds_binary)
        # print("======>")
        # print(list(zip("tn fp fn tp".split(), confusion_matrix(y_trues_binary, y_preds_binary).ravel())))
        # print(roc_score)
        # print(precision, recall, f1_score)
        # print(acc)
        # top1-top5, acc, auc, precision, recall, f1
        
        # 转为二分类。
        # 所有节点预测是否是
        
        result = {
        "top1":top_dict["top1"],
        "top2":top_dict["top2"],
        "top3":top_dict["top3"],
        "top4":top_dict["top4"],
        "top5":top_dict["top5"],
        "recall": recall,
        "precision": precision,
        "f1": f1_score,
        "acc": acc,
        "auc":roc_score
        # "eval_threshold":best_threshold,
        }
        
        return result

def roc_curve_point(y_score, y_true, classifier_nums):
    logits = y_score.detach().cpu().numpy()
    y_trues = y_true.detach().cpu().numpy()
    
    y_trues_binary = label_binarize(y_trues, classes=range(classifier_nums))
    y_trues_binary=np.reshape(y_trues_binary, (-1))
    logits_binary = np.reshape(logits, (-1))
    fpr, tpr, thresholds = roc_curve(y_trues_binary, logits_binary, pos_label=1)
    
    
    return {"roc":{"fpr":fpr.tolist(), "tpr":tpr.tolist(), "thresholds":thresholds.tolist()}}

def top_5(y_score, y_true):
    #print(y_score)
    #print(y_score.size())
    #print(y_true)
    #print(y_true.size())
    top_5 = [0, 0, 0, 0, 0]

    _, maxk = torch.topk(y_score, 5, dim=-1)
    total = y_true.size(0)
    test_labels = y_true.view(-1, 1)  # reshape labels from [n] to [n,1] to compare [n,k]

    for i in range(5):
        top_5[i] += (test_labels == maxk[:, 0:i + 1]).sum().item()

    return {("top" + str(i + 1)): top_5[i] / total for i in range(5)}


def cal_early_stopping_metric(task_metric_results: List[Dict[str, np.ndarray]], ) -> float:
    # Early stopping based on accuracy; as we are trying to minimize, negate it:
    acc = sum([m['top1'] for m in task_metric_results]) / float(len(task_metric_results))
    return -acc


def pretty_print_epoch_task_metrics(task_metric_results: List[Dict[str, np.ndarray]], num_graphs: int,
                                    num_batchs: int) -> str:
    # 这里接收cal_metrics返回的metrics，并指定成字符串输出。
    # top5 = [[] for _ in range(5)]
    # for i in range(5):
    #     top5[i] = sum([m['top' + str(i + 1)] for m in task_metric_results]) / float(len(task_metric_results))

    # top_str = " ".join(["Top" + str(i + 1) + " %.3f" % top5[i] for i in range(5)])
    
    metrics_mean_dict = dict()
    for key in task_metric_results[0].keys():
        metrics_mean_dict[key] = sum([m[key] for m in task_metric_results]) / float(len(task_metric_results))
    
    metrics_str = " ".join([f"{key} {round(metrics_mean_dict[key], 4)}" for key in metrics_mean_dict.keys()])
    
    return metrics_str


def average_weights(model_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    w_avg = copy.deepcopy(model_weights[0])
    for key in w_avg.keys():
        for i in range(1, len(model_weights)):
            w_avg[key] += model_weights[i][key]
        w_avg[key] = torch.div(w_avg[key], len(model_weights))
    return w_avg