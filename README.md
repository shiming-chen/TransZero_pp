# TransZero [[arXiv]](https://arxiv.org/pdf/2112.01683.pdf)


This repository contains the testing code for the paper  "***TransZero++: Cross Attribute-guided Transformer for Zero-Shot Learning***" submitted to TPAMI. 
**We will release all codes of this work later**.



## Preparing Dataset and Model

We provide trained models ([Google Drive](https://drive.google.com/drive/folders/1WK9pm2eX2Rl4rWqXqe_EZiAM8wWB8yqG?usp=sharing)) on three different datasets: [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [SUN](http://cs.brown.edu/~gmpatter/sunattributes.html), [AWA2](http://cvml.ist.ac.at/AwA2/) in the CZSL/GZSL setting. You can download model files as well as corresponding datasets, and organize them as follows: 
```
.
├── saved_model
│   ├── TransZero_CUB_CZSL.pth
│   ├── TransZero_CUB_GZSL.pth
│   ├── TransZero_SUN_CZSL.pth
│   ├── TransZero_SUN_GZSL.pth
│   ├── TransZero_AWA2_CZSL.pth
│   └── TransZero_AWA2_GZSL.pth
├── data
│   ├── CUB/
│   ├── SUN/
│   └── AWA2/
└── ···
```

## Requirements
The code implementation of **TransZero** mainly based on [PyTorch](https://pytorch.org/). All of our experiments run and test in Python 3.8.8. To install all required dependencies:
```
$ pip install -r requirements.txt
```
## Runing
Runing following commands and testing **TransZero** on different dataset:

CUB Dataset: 
```
$ python test.py --config config/CUB_CZSL.json      # CZSL Setting
$ python test.py --config config/CUB_GZSL.json      # GZSL Setting
```
SUN Dataset:
```
$ python test.py --config config/SUN_CZSL.json      # CZSL Setting
$ python test.py --config config/SUN_GZSL.json      # GZSL Setting
```
AWA2 Dataset: 
```
$ python test.py --config config/AWA2_CZSL.json     # CZSL Setting
$ python test.py --config config/AWA2_GZSL.json     # GZSL Setting
```

## Results
Results of our released models using various evaluation protocols on three datasets, both in the conventional ZSL (CZSL) and generalized ZSL (GZSL) settings.

| Dataset | Acc(CZSL) | U(GZSL) | S(GZSL) | H(GZSL) |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| CUB | 78.6 | 67.5 | 73.6 | 70.3 |
| SUN | 67.6 | 48.2 | 37.6 | 42.3 |
| AWA2 | 72.6 | 64.5 | 82.7 | 72.5 |

**Note**: All of above results are run on a server with an AMD Ryzen 7 5800X CPU and a NVIDIA RTX A6000 GPU.


## References
Parts of our codes based on:
* [hbdat/cvpr20_DAZLE](https://github.com/hbdat/cvpr20_DAZLE)
* [zhangxuying1004/RSTNet](https://github.com/zhangxuying1004/RSTNet)

## Contact
If you have any questions about codes, please don't hesitate to contact us by gchenshiming@gmail.com or hoongzm@gmail.com.
