# MLLA for Object Detection

Our experiments are conducted on COCO datast based on [mmdetection](https://github.com/open-mmlab/mmdetection).

## Results and Models

### MaskRCNN

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MLLA-T | ImageNet-1K | 1x | 46.8 | 42.1 | 44M | 255G | [config](configs/mlla/mlla_t_mrcnn_1x.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/ce362e868fb5491ca55c/?dl=1) |
| MLLA-T | ImageNet-1K | 3x | 48.8 |   43.8   | 44M | 255G | [config](configs/mlla/mlla_t_mrcnn_3x.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/339d49127565432fb912/?dl=1) |
| MLLA-S | ImageNet-1K | 1x | 49.2 | 44.2 | 63M | 319G | [config](configs/mlla/mlla_s_mrcnn_1x.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/3d29bec850e0441a8842/?dl=1) |
| MLLA-S | ImageNet-1K | 3x | 50.5 | 44.9 | 63M | 319G | [config](configs/mlla/mlla_s_mrcnn_3x.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/b501c886982e4c7f926d/?dl=1) |
| MLLA-B | ImageNet-1K | 1x | 50.5 | 45.0 | 115M | 502G | [config](configs/mlla/mlla_b_mrcnn_1x.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/826278b01bed4df2bbc1/?dl=1) |

## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation and dataset preparation.

### ImageNet-1K Pretrained Model

Please download the ImageNet-1K pretrained models and place them under `./data/` folder, e.g. `./data/MLLA_T.pth`.

### Inference

```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

### Training

To train a detector with pre-trained models, run:
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

