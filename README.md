# GTCN

The Implementation of paper "Toward Interpretable Graph Tensor Convolution Neural Network for Code Semantics Embedding"

In this paper, we propose a novel interpretable model, called **graph tensor convolution neural network (GTCN)**, to generate accurate code embedding, which is capable of comprehensively capturing the distant information of code sequences and rich code semantics structural informa.

## 1. Datasets

### 1.1 Using existing datasets

Note that due to the space limitations of Github, we cannot upload all datasets. This will drastically change the results of the experiment. All our datasets come from public papers. For more accurate experiments, we strongly recommend that you download the datasets according to the following url and set them:

CSharp: https://aka.ms/iclr18-prog-graphs-dataset

Python: https://www.sri.inf.ethz.ch/research/plml

You can follow the next steps to get the fully dataset:

1. Download the two datasets
2. put the corresponding file into: ```data/$dataset name$/$train data$/raw```. You can divide them into training, validation, testing randomly.
3. Run the ```single_task.py``` file and specify the language type of the dataset. The file will call the appropriate file in the dataProcessing folder and process the code into a format acceptable to the model.

Example:
```python single_task.py --output_model=vm --train_data_dir=your_data_dir --validate_data_dir=your_data_dir --dataset_name=csharp```

When you process the dataset for the first time, the following instructions are displayed:

```======data preprocessing```

When the dataset is successfully processed, the following instructions are displayed: 

```======data processing done!```


The dataset only needs to be processed once and a cache file will be generated and placed under the dataset. When the model is run again later, no further processing is required. Note that if you need to add new data, remember to delete these cache folders and run the above process again.

Notes: Since the python dataset is too large compared to csharp, for more balanced training, we recommend that you limit the python dataset to 10,000 training data and 2,000 test data.

### 1.2 Compile self code to datasets

Due to the size limitation of the repository, we can't upload all the data. We publish the data generation program that anyone can use to compile his own dataset.

The program is  modified from microsoft to compile the existing code into the format needed for model training, which is published at: [shandianchengzi
/graph-based-code-modelling](https://github.com/shandianchengzi/graph-based-code-modelling/tree/shan).
Compilation for python and C# code is now well supported.

## 2. Training

The model supports various downstream tasks including: Variable misuse detection(VM), Code Completion(CC), Vulnerability detection. We can switch the task by specifing the arguments.

### 2.1 Variable Misuse

Specify output_model=vm can choose the variable misuse task.

Example:
```python
python single_task.py --backbone_model tensor_gcn --output_model=vm --train_data_dir=your_data_dir --validate_data_dir=your_data_dir --dataset_name=csharp  --result_dir=./result
```

### 2.2 Code Completion

Specify output_model=cc can choose the variable misuse task.

Example:
```python
python single_task.py --backbone_model tensor_gcn --output_model=cc --train_data_dir=your_data_dir --validate_data_dir=your_data_dir --dataset_name=python  --result_dir=./result
```

### 2.3 Detect Vulnerability Code

Due to the different data sets for vulnerability detection. Please refer to this repository for more details: https://github.com/wannch/VDoTR.


## 3. Evaluate

Use evaulate.py for evaluating the model preformance.

set roc=True can return the roc points for plot.
set case_analysis can return the probability of each candidate in vm and cc.

Example:
```python
python evaluate.py --evaulate_data_idr=your_eva_data_dir --dataset_name=csharp --load_model_file=your_model_checkpoint --roc=True --case_analysis=True
```
## 4. Requirements

* torch >= 1.8.0
* torch-geometric  == 1.7.0
* python >= 3.7
* numpy >= 1.19
* tqdm >= 4.56.1

Notes that torch_geometric is updated frequently. Too high a version will cause incompatibility.

We may have overlooked some other dependencies, please install the latest version of the missing package directly.
