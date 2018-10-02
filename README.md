# Semantic Segmentation using U-Net

Pytorch implementation of [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) for segmentation of aerial maps into google maps.

## Dataset

I used the pix2pix maps dataset available over [here](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/)

## Requirements

* [pytorch](https://pytorch.org/): `0.4.1`
* [torchvision](https://pytorch.org/docs/stable/torchvision/index.html): `0.2.1`

P.S: It took 12 hours to train on a 1050Ti with a batch size of 5 for 100 epochs. If I tried to increase the batch size, I ran out of memory. I asked around and people suggested using [checkpoint](https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint) and I found a [discussion post](https://discuss.pytorch.org/t/torch-utils-checkpoint-checkpoint/16827) related to it. Haven't tried it yet, therefore any suggestion or crtiques are always welcome. 
