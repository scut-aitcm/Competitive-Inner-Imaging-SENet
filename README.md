# CompetitiveSENet
---

Source code of paper: 

   **Competitive Inner-Imaging Squeeze and Excitation for Residual Network** ([https://arxiv.org/abs/1807.08920](https://arxiv.org/abs/1807.08920))


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

## Requirements

- **MXNet 1.2.0**
- Python 2.7
- CUDA 8.0+(for GPU)

## Citation

>@article{hu2018competitive,  
>  title={Competitive Inner-Imaging Squeeze and Excitation for Residual Network},  
>  author={Hu, Yang and Wen, Guihua and Luo, Mingnan and Dai, Dan},  
>  journal={arXiv preprint arXiv:1807.08920},  
>  year={2018}  
>}

## Results
Best record of this novel model on CIFAR-10 and CIFAR-100 (used "*mixup*" ([https://arxiv.org/abs/1710.09412](https://arxiv.org/abs/1710.09412))) can achieve: **97.55%** and **84.38%**.
 
The test result on Kaggle: [CIFAR-10 - Object Recognition in Images](https://www.kaggle.com/c/cifar-10) 

![](pictures/cifar10_kaggle.png)

