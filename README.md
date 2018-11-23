# CompetitiveSENet
---

Source code of paper: 

   **(not availbale now)** 

---
## Architecture

|Competitive Squeeze-Exciation Architecutre for Residual block|
|-|
|![architecutre](pictures/architecture.png)|

---

SE-ResNet module and CMPE-SE-ResNet modules:

|Normal SE|Double FC squeezes|Conv 2x1 pair-view|Conv 1x1 pair-view|
|-|-|-|-|
|![](pictures/se_resnet_module.png)|![](pictures/cmpe_se_resnet_double_FC_squeeze.png)|![](pictures/cmpe_se_resnet_conv2x1.png)|![](pictures/cmpe_se_resnet_conv1x1.png)|

The Novel Inner-Imaging Mechanism for Channel Relation Modeling in Channel-wise Attention of ResNets (even All CNNs):

|Basic Inner-Imaing Mode|Folded Inner-Imaging Mode|
|-|-|
|![](pictures/Basic-Inner-Imaging.png)|![](pictures/Folded-Inner-Imaging.png)|

---

## Requirements

- **MXNet 1.2.0**
- Python 2.7
- CUDA 8.0+(for GPU)

---

## Citation

not available now

---

## Essential Results
Best record of this novel model on CIFAR-10 and CIFAR-100 (used "*mixup*" ([https://arxiv.org/abs/1710.09412](https://arxiv.org/abs/1710.09412))) can achieve: **97.55%** and **84.38%**.
 
The test result on Kaggle: [CIFAR-10 - Object Recognition in Images](https://www.kaggle.com/c/cifar-10) 

![](pictures/cifar10_kaggle.png)

Inner-Imaging Examples & Channel-wise Attention Outputs

![](pictures/appendix_a_fig1.png)

