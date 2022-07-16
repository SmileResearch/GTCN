# GTCN

## Running

To train a model, it suffices to run ```python single_task.py --backbone_model tensor_gcn```, for example as follows:
```python
$ python single_task.py --backbone_model  tensor_gcn
```
## Detect Vulnerability Code

please refer to https://github.com/wannch/VDoTR

## Datasets

Note that due to the space limitations of Github, we cannot upload all datasets. This will drastically change the results of the experiment. All our datasets come from public papers. For more accurate experiments, we strongly recommend that you download the datasets according to the following url and set them:

CSharp: https://aka.ms/iclr18-prog-graphs-dataset

Python: https://www.sri.inf.ethz.ch/research/plml

Download the two datasets and put the corresponding file into: ```data/$dataset name$/$train_data or validate_data$/raw ```.

Since the python dataset is too large compared to csharp, for more balanced training, we recommend that you limit the python dataset to 10,000 training data and 2,000 test data.

## Requirements

* torch >= 1.8.0
* torch-geometric  >= 1.7.0
* python >= 3.7
* numpy >= 1.19
* tqdm >= 4.56.1

We may have overlooked some other dependencies, please install the latest version of the package directly.
