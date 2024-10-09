# GURPP

Overview
----------------------------------
Official code for: **Urban Region Embedding Pre-training and Prompting: A Graph-based Approach**

[//]: # (More details to come after accepted.)

## Framework
<img src='img_1.png' alt="framework" width="600">

## Requirements
- python==3.8.8
- pytorch==1.13.1
- numpy==1.23.5
- dgl==2.2.1
- scikit-learn==1.2.0
- pandas==1.5.2

## Running
to run the code, you can run the following command:

1. train pre-training model
```bash
python train_gurp.py
```
2. train task-learnable prompt model
```bash
python train_gurp_prompt.py
```

to reproduce the results in the paper, you can run the following command:
```bash
python reproduce.py
