# MLLA for Semantic Segmentaion

Our experiments are conducted on ADE20K dataset based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/).

## Results and Models

### UperNet

| Backbone | Pretrain | Lr Schd | mIoU (SS) | mIoU (MS) | #params | FLOPs | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MLLA-B | ImageNet-1K | 160K | 51.9 | 52.5 | 128M | 1183G | [config](configs/mlla/mlla_b_upernet.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/41760d4a531a4ffc9cbf/?dl=1) |

## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation) for installation and dataset preparation.

### ImageNet-1K Pretrained Model

Please download the ImageNet-1K pretrained models and place them under `./data/` folder, e.g. `./data/MLLA_T.pth`.

### Inference

```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval mIoU

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval mIoU
```

### Training

To train with pre-trained models, run:
```
# multi-gpu training
torchrun --nproc_per_node <GPU_NUM> tools/train.py <CONFIG_FILE> --launcher="pytorch"
```

## Citation

If you find this repo helpful, please consider citing us.

```latex
@inproceedings{han2024demystify,
  title={Demystify Mamba in Vision: A Linear Attention Perspective},
  author={Han, Dongchen and Wang, Ziyi and Xia, Zhuofan and Han, Yizeng and Pu, Yifan and Ge, Chunjiang and Song, Jun and Song, Shiji and Zheng, Bo and Huang, Gao},
  booktitle={NeurIPS},
  year={2024},
}
```
