import torch
import numpy as np
from typing import Any, Dict, Optional, Tuple, List, Iterable

def cal_metrics(y_score, y_true):
    with torch.no_grad():
        metrics = {}
        #print(y_score)
        #print(y_true)
        #y_score_softmax = nn.functional.softmax(y_score, dim=-1)
        #print(y_score_softmax)
        #y_score_softmax_np = y_score_softmax.cpu().detach().numpy()
        #y_true_np = y_true.cpu().detach().numpy()

        top5 = top_5(y_score, y_true)
        metrics.update(top5)
        return metrics

def top_5(y_score, y_true):
    #print(y_score)
    #print(y_score.size())
    #print(y_true)
    #print(y_true.size())
    top_5 = [0, 0, 0, 0, 0]

    _, maxk = torch.topk(y_score, 5, dim=-1)
    total = y_true.size(0)
    test_labels = y_true.view(
        -1, 1)  # reshape labels from [n] to [n,1] to compare [n,k]

    for i in range(5):
        top_5[i] += (test_labels == maxk[:, 0:i + 1]).sum().item()

    return {("top" + str(i + 1)): top_5[i] / total for i in range(5)}

def cal_early_stopping_metric(
    task_metric_results: List[Dict[str, np.ndarray]],
) -> float:
    # Early stopping based on accuracy; as we are trying to minimize, negate it:
    acc = sum([m['top1'] for m in task_metric_results]) / float(
        len(task_metric_results))
    return -acc

def pretty_print_epoch_task_metrics(task_metric_results: List[Dict[
    str, np.ndarray]], num_graphs: int, num_batchs: int) -> str:
    top5 = [[] for _ in range(5)]
    for i in range(5):
        top5[i] = sum([m['top' + str(i + 1)] for m in task_metric_results
                        ]) / float(len(task_metric_results))

    top_str = " ".join(
        ["Top" + str(i + 1) + " %.3f" % top5[i] for i in range(5)])
    return top_str