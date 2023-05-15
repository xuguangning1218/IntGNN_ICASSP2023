# IntGNN

Official source code for paper [Int-GNN: a User Intention Aware Graph Neural Network for Session-Based Recommendation](https://ieeexplore.ieee.org/document/10097031) accepted by ICASSP 2023

### Overall Architecture of IntGNN
![image](https://github.com/xuguangning1218/IntGNN_ICASSP2023/blob/master/figure/model.jpg)

### Environment Setting
```
pytorch == 1.12.0
numpy == 1.20.3
tqdm == 4.61.2
torchvision == 0.13.0
```  

###  Source Files Description

```
-- datasets # dataset folder
  -- diginetica # diginetica dataset 
  -- retailRocket_DSAN # retail dataset
  -- Tmall # Tmall dataset
-- figure # figure provider
  -- model.jpg # architecture of Int-GNN model 
-- pytorch_code # main code of the project
  -- utils # the utils file folder
  -- models # the models file folder
  -- controller.py # the basic control operation
  -- train_intgnn.py # the core code of the Int-GNN
```

### Run

When the environment and datasets are cloned, you can train the IntGNN by running the following code:

```
cd ./pytorch_code
python train_intgnn.py
```

### Citation
If you find this code or idea useful, please cite our work:
```bib
@INPROCEEDINGS{xu2023Int,
  author={Xu, Guangning and Yang, Jinyang and Guo, Jinjin and Huang, Zhichao and Zhang, Bowen},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Int-GNN: A User Intention Aware Graph Neural Network for Session-Based Recommendation}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10097031}}
```
