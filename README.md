# CompetitiveSENet
---

Source code of paper: 

**Competitive Inner-Imaging Squeeze and Excitation for Residual Network** ([https://arxiv.org/abs/1807.08920](https://arxiv.org/abs/1807.08920))


---
## Architecture

|Competitive Squeeze-Exciation Architecutre for Residual block|
|-|
|<img src="pictures/fig1.png", width="1000">|

---

SE-ResNet module and CMPE-SE-ResNet modules:

|Normal SE|Double FC squeezes|Conv 2x1 pair-view|Conv 1x1 pair-view|
|-|-|-|-|
|<img src="pictures/se_resnet_module.png", width="250">|<img src="pictures/cmpe_se_resnet_double_FC_squeeze.png", width="250">|<img src="pictures/cmpe_se_resnet_conv2x1.png", width="250">|<img src="pictures/cmpe_se_resnet_conv2x1.png", width="250">|

## Requirements

- mxnet.gluon

## Results
Best record of this novel model on CIFAR-10 and CIFAR-100 (used "*mixup*" ([https://arxiv.org/abs/1710.09412](https://arxiv.org/abs/1807.08920))) can achieve: **97.55%** and **84.38%**.

Error rates(%) of Wide Residual Networks and pre-act ResNets on CIFAR10/100:

|WRN|Resnet|
|-|-|
|<img src="pictures/cmpe_se_wrn_table.png", width="1000">|<img src="pictures/cmpe_se_resnet.png", width="1000">|

More results can be found in our paper: [arXiv paper: Competitive Inner-Imaging Squeeze and Excitation for Residual Network](https://arxiv.org/abs/1807.08920)

## Notes:
The result of our best model in Kaggle competition : [CIFAR-10 - Object Recognition in Images](https://www.kaggle.com/c/cifar-10) 

<img src="pictures/cifar10_kaggle.png", width="1000">

