
## Masked Jigsaw Puzzle: A Versatile Position Embedding for Vision Transformers, CVPR2023

![Python 3.6](https://img.shields.io/badge/python-3.6.8-green.svg)
![Packagist](https://img.shields.io/badge/Pytorch-1.6.0-red.svg)
![Last Commit](https://img.shields.io/github/last-commit/yhlleo/MJP)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)]((https://github.com/yhlleo/MJP/graphs/commit-activity))
![Contributing](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)


**[[arXiv]](https://arxiv.org/pdf/2205.12551.pdf) | [[Codes]](https://github.com/yhlleo/MJP)** <br> 
[Bin Ren](https://scholar.google.com/citations?hl=en&user=Md9maLYAAAAJ)<sup>1,2</sup>$^\star$, [Yahui Liu](https://scholar.google.com/citations?user=P8qd0rEAAAAJ&hl=en)<sup>2</sup>$^\star$, [Yue Song](https://scholar.google.com/citations?hl=en&user=Uza2i10AAAAJ)<sup>2</sup>, [Wei Bi](https://scholar.google.com/citations?user=aSJcgQMAAAAJ&hl=en)<sup>3</sup>, [Rita Cucchiara](https://scholar.google.com/citations?user=OM3sZEoAAAAJ&hl=en)<sup>4</sup>, [Nicu Sebe](https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl=en)<sup>2</sup> and [Wei Wang](https://scholar.google.com/citations?user=k4SdlbcAAAAJ&hl=en)<sup>5</sup>$^\dagger$ <br>
$\star$: Equal Contribution, $\dagger$: Corresponding Author <br>
<sup>1</sup>University of Pisa, Italy, <br>
<sup>2</sup>University of Trento, Italy, <br> 
<sup>3</sup>Tencent AI Lab, China, <br>
<sup>4</sup>University of Modena and Reggio Emilia, Italy, <br>
<sup>5</sup>Beijing Jiaotong University, China <br>


<p align="center">
<img src="figures/overview.jpg" width="800px"/>
<br>
The main idea of random jigsaw shuffle algorithm and the overview the proposed MJP.
</p>

The repository offers the official implementation of our paper in PyTorch.

:t-rex:News(March 4, 2023)!  **Our paper is accepted by CVPR2023!**


### Abstract
Position Embeddings (PEs), an arguably indispensable component in Vision Transformers (ViTs), have been shown to improve the performance of ViTs on many vision tasks. However, PEs have a potentially high risk of privacy leakage since the spatial information of the input patches is exposed. This caveat naturally raises a series of interesting questions about the impact of PEs on accuracy, privacy, prediction consistency, \etc. To tackle these issues, we propose a Masked Jigsaw Puzzle (MJP) position embedding method. In particular, MJP first shuffles the selected patches via our block-wise random jigsaw puzzle shuffle algorithm, and their corresponding PEs are occluded. Meanwhile, for the non-occluded patches, the PEs remain the original ones but their spatial relation is strengthened via our dense absolute localization regressor. The experimental results reveal that 1) PEs explicitly encode the 2D spatial relationship and lead to severe privacy leakage problems under gradient inversion attack; 2) Training ViTs with the naively shuffled patches can alleviate the problem, but it harms the accuracy; 3) Under a certain shuffle ratio, the proposed MJP not only boosts the performance and robustness on large-scale datasets (\emph{i.e.}, ImageNet-1K and ImageNet-C, -A/O) but also improves the privacy preservation ability under typical gradient attacks by a large margin. 

### 中文简介：
**MJP: 一种用于视觉Transformer的多才多艺（多功能）的位置编码（PEs）方法：**

众所周知，Transformer中的多头注意力机制自身对输入信息的位置顺序并不敏感，但对于视觉任务，物体之间的2D空间位置关系至关重要（我们也在本文中首次显性地验证了PE确实能够学到2D空间关系），因此PEs已经默认是视觉Transformer中不可或缺的一部分。但我们发现PEs在对ViT提供正面贡献的同时，也让Transformer付出了代价。例如，让注意力机制固有的位置不敏感属性受到挑战，具体而言，当模型在面对同样输入内容但不同顺序（一致性挑战）的时候，就显得力不从心（比如分类预测准确性降低）。此外，PEs还会使得Transformer在面对联邦学习中梯度攻击的时候，去主动“交代”自己学到的位置信息，从而使得私有数据能被轻而易举的恢复，造成隐私泄漏等安全性问题。因此，本文中提出了一种“多才多艺”的PEs方法：MJP，MJP首先随机选中输入图像块中一定比例的小块并打乱其空间顺序，且对应的PEs会被遮挡。对于未被打乱的部分，则保持他们原有的PEs，与此同时，我们以自监督回归的方式，对这些块施加位置先验信息。实验结果表明，MJP使得Transformer在面对分类预测准确性，一致性，以及安全性等共同挑战的时候，能够维持一个多方受益的均衡局面。具体而言，在不伤害Transformer常规分类预测准确性的同时（甚至提高了），模型的一致性和稳健性也得到大幅度提升。此外，令我们的惊喜的是，配备了MJP的Transformer，在面对典型的梯度攻击的时候，也能极大程度的提升模型的隐私保护能力。更为详细的信息请参考我们的论文：https://arxiv.org/abs/2205.12551 和代码：https://github.com/yhlleo/MJP, 也欢迎大家一起交流讨论。


### Datasets

|Dataset|Download Link|
|:-----|:-----|
|[ImageNet](https://www.image-net.org/)|[train](http://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar),[val](http://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar)|


 - Download the datasets by using codes in the `scripts` folder.

```
dataset_name
  |__train
  |    |__category1
  |    |    |__xxx.jpg
  |    |    |__...
  |    |__category2
  |    |    |__xxx.jpg
  |    |    |__...
  |    |__...
  |__val
       |__category1
       |    |__xxx.jpg
       |    |__...
       |__category2
       |    |__xxx.jpg
       |    |__...
       |__...
```

### Checkpoints 
You can find our pretrained checkpoints and 999 images sampled from ImageNet for attacking [here](https://drive.google.com/drive/folders/1P6LnqhLTyG7CRcb7_NsFDVmQ1cOFoNeV?usp=sharing).

### Training 

After prepare the datasets, we can simply start the training with 8 NVIDIA V100 GPUs:

```
$ sh train.sh
```

### Evaluation 

 - Accuracy on Masked Jigsaw Puzzle

```
$ python3 eval.py 
```

 - Consistency on Masked Jigsaw Puzzle

```
$ python3 consistency.py
```

 - Evaluations on image reconstructions

See the codes [`MSE`](./eval/cal_mse.py), [`PSNR/SSIM`](./eval/cal_psnr_ssim.py), [`FFT2D`](./eval/cal_fft2d.py), [`LPIPS`](cal_lpips.py).

### Gradient Attack
<p align="center">
<img src="figures/gradient_attack_cvpr.jpg" width="800px"/>
<br>
Visual comparisons on image recovery with gradient attacks.
</p>

We refer to the public repo: [JonasGeiping/breaching](https://github.com/JonasGeiping/breaching).

### Acknowledgement

This repo is built on several existing projects:

 - [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
 - [BeiT](https://github.com/microsoft/unilm/tree/master/beit)
 - [VTs-Drloc](https://github.com/yhlleo/VTs-Drloc)

### Citation

If you take use of our code or feel our paper is useful for you, please cite our papers:

```
@article{ren2023masked,
    author    = {Ren, Bin and Liu, Yahui and Song, Yue and Bi, Wei and and Cucchiara, Rita and Sebe, Nicu and Wang, Wei},
    title     = {Masked Jigsaw Puzzle: A Versatile Position Embedding for Vision Transformers},
    journal   = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2023}
}
```

If you have any questions, please contact me without hesitation (yahui.cvrs AT gmail.com or bin.ren AT unitn.it).

