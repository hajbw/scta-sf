# SCTA-SF: Slim Cross Transfer Attention with Statistical Fusion for Few-Shot Image Classification

This repo is the pytorch implementation of SCTA-SF: Slim Cross Transfer Attention with Statistical Fusion for Few-Shot Image Classification

## Abstract

Few-shot image classification is a challenging problem in computer vision, aiming to distinguish different classes only based on few labeled support samples. However, it is difficult to achieve precise and robust sample-level interactions. Besides, the existing methods usually suffer from high computational complexity. To solve these challenging problems, we present a SCTA-SF framework to achieve efficient sample-level interactions and intelligent feature augmentation simultaneously. First, we propose a Slim Cross Transfer Attention (SCTA) module to establish reliable sample-level interactions between the support and query sets using a compress-cross polar axial mechanism, which greatly reduces the computational complexity while maintaining the performance of the network. Second, we propose a Statistical Fusion (SF) generator to synthesize new samples’ features. Our SF generator employs a learnable direction relation matrix to eliminate outliers, ensuring the generation of robust samples with a reasonable statistical distribution. At last, multiple comparison experiments were conducted on several widely-recognized few-shot learning datasets (“MiniImageNet”, “TieredImageNet” and “CUB”). The experimental results demonstrate that our method achieves the state-of-the-art performance, achieving the highest accuracy of 74.13% (1-shot) and 89.62% (5-shot) on the widely used “MiniImageNet” dataset.

## Standard Few-shot Classification Results

| Dataset        | 5-way-1-shot | 5-way-5-shot |
| -------------- | :----------: | :----------: |
| miniImageNet   |  74.13±0.48  |  89.62±0.40  |
| tieredImageNet |  79.23±0.55  |  93.14±0.26  |
| CUB            |  87.26±0.38  |  95.80±0.19  |

## Setup

create a Python environment with the following dependencies:

```shell

torch
torchvision
tqdm
tensorboard
scikit-image
scikit-learn
scipy
numpy
collections
einops
Pillow

```

Clone the repository.

```shell

git clone https://github.com/hajbw/scta-sf
cd ./scta-sf

```

Prepare the datasets `miniImageNet`, `tieredImageNet`, and `CUB`.

Download the checkpoints:

follow this link to [BaiduNetDisk](https://pan.baidu.com/s/1th4oOXEuVfii2QepRNwAKw) (extraction code:y4v2) and download the corresponding checkpoint.

| dataset      | task         | path                                                      |
| ------------ | ------------ | --------------------------------------------------------- |
| miniImageNet | 5-way-1-shot | ResNet12_meta_deepbdc_5way_1shot_metatrain/best_model.tar |
| miniImageNet | 5-way-5-shot | ResNet12_meta_deepbdc_5way_5shot_metatrain/best_model.tar |

**NOTE**: We are re-testing our checkpoints for better reproducibility, and checkpoints for other datasets will be released soon.

## Testing

Configure the `run_test.sh` and run the code using the following command:

```shell

sh ./scripts/<dataset name>/run_test.sh

```

## Acknowledgment

We would like to thank the following repositories for providing useful components in our work.

- [DeepBDC](https://github.com/Fei-Long121/DeepBDC)
- [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot)
- [RFS](https://github.com/WangYueFt/rfs/)
