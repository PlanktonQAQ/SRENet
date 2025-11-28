# SRENet: Saliency-Based Lighting Enhancement Network

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/PlanktonQAQ/SRENet)
[![Paper Link](https://img.shields.io/badge/View-Paper-orange)](https://ieeexplore.ieee.org/abstract/document/11080243)

## Abstract
Lighting enhancement is a classical topic in low-level image processing. Existing studies mainly focus on global illumination optimization while overlooking local semantic objects, and this limits the performance of exposure compensation. In this paper, we introduce SRENet, a novel lighting enhancement network guided by saliency information. It adopts a two-step strategy of foreground-background separation optimization to achieve a balance between global and local illumination. In the first step, we extract salient regions and implement the local illumination enhancement that ensures the exposure quality of salient objects. Next, we utilize a fusion module to process global lighting optimization based on local enhanced results. With the two-step strategy, the proposed SRENet yield better lighting enhancement for local illumination while preserving the globally optimal results. Experimental results demonstrate that our method obtains more effective enhancement results for various tasks of exposure correction and lighting quality improvement. The source code and pre-trained models are available at https://github.com/PlanktonQAQ/SRENet.

## Authors
- [Yuming Fang](http://sim.jxufe.cn/JDMKL/ymfang_EN.html)
- Chen Peng
- [Chenlei Lv*](https://aliexken.github.io) (Corresponding Author)
- [Weisi Lin](https://personal.ntu.edu.sg/wslin/Home.html)

## Resources
| Resource | Link |
|----------|------|
| Source Code & Pre-trained Models | [GitHub Repository](https://github.com/PlanktonQAQ/SRENet) |
| Checkpoint Files | [OneDrive](https://1drv.ms/f/c/dc8b31693367a205/Etg1MTiqEydJjQGjXTfYquABMUbkd5a6TB5XhWkc_kwdCA?e=P5vyUw) |
| Test Dataset | [OneDrive](https://1drv.ms/f/c/dc8b31693367a205/EmE-f66p9QZCkupklsF1r48BnTRlAj8S5wsb-ITpz3UaRw?e=jkJe0v) |


## Getting Started
### Prerequisites
- Configure the environment using `environment.yml`:
  ```bash
  conda env create -f environment.yml
  conda activate chen_torch

## Contact
Feel free to contact me if there is any question (ChenPeng Ispengchen@outlook.com).

