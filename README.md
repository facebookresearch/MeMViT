## MeMViT: Memory-Augmented Multiscale Vision Transformer for Efficient Long-Term Video Recognition

<p align="center">
  <img width="595" src="https://user-images.githubusercontent.com/1841547/172720371-8403cf48-c2b3-4d28-b624-00edc7b0bf97.png">
</p>

This is a PyTorch implementation of [the MeMViT paper](https://arxiv.org/abs/2201.08383) (CVPR 2022 oral):
```
@inproceedings{memvit2022,
  title={{MeMViT: Memory-Augmented Multiscale Vision Transformer for Efficient Long-Term Video Recognition}},
  author={Wu, Chao-Yuan and Li, Yanghao and Mangalam, Karttikeya and Fan, Haoqi and Xiong, Bo and Malik, Jitendra and Feichtenhofer, Christoph},
  booktitle={CVPR},
  year={2022}
}
```
MeMViT builds on the MViT models:
```
@inproceedings{li2021improved,
  title={{MViTv2}: Improved multiscale vision transformers for classification and detection},
  author={Li, Yanghao and Wu, Chao-Yuan and Fan, Haoqi and Mangalam, Karttikeya and Xiong, Bo and Malik, Jitendra and Feichtenhofer, Christoph},
  booktitle={CVPR},
  year={2022}
}
@inproceedings{fan2021multiscale,
  title={Multiscale vision transformers},
  author={Fan, Haoqi and Xiong, Bo and Mangalam, Karttikeya and Li, Yanghao and Yan, Zhicheng and Malik, Jitendra and Feichtenhofer, Christoph},
  booktitle={ICCV},
  year={2021}
}
```

## Model checkpoints
On the AVA dataset:
| name | mAP | #params (M) | GFLOPs | pre-train model | model |
|:---:|:---:|:---:|:---:| :---:| :---:|
| MeMViT-16, 16x4 | 29.3 | 35.4 | 58.7 | [K400-pretrained model](https://dl.fbaipublicfiles.com/memvit/models/Kinetics/MeMViT_16L_16x4_K400.pyth) | [model](https://dl.fbaipublicfiles.com/memvit/models/AVA/MeMViT_16L_16x4_K400pt.pyth) |
| MeMViT-24, 32x3 | 32.3 | 52.6 | 211.7 | [K600-pretrained model](https://dl.fbaipublicfiles.com/memvit/models/Kinetics/MeMViT_24L_32x3_K600.pyth) | [model](https://dl.fbaipublicfiles.com/memvit/models/AVA/MeMViT_24L_32x3_K600pt.pyth) |
| MeMViT-24, 32x3 | 34.4 | 52.6 | 211.7 | [K700-pretrained model](https://dl.fbaipublicfiles.com/memvit/models/Kinetics/MeMViT_24L_32x3_K700.pyth) | [model](https://dl.fbaipublicfiles.com/memvit/models/AVA/MeMViT_24L_32x3_K700pt.pyth) |


## Installation
This repo is a modification on the [PySlowFast repo](https://github.com/facebookresearch/SlowFast). Installation and preparation follow that repo.

## Training
Please modify the data paths and pre-training checkpoint path in config file accordingly and run, e.g.,
```
python tools/run_net.py \
  --cfg configs/AVA/MeMViT_16_K400.yaml \
```

## Evaluation
To evaluate a pretrained MeMViT model:
```
python tools/run_net.py \
  --cfg configs/AVA/MeMViT_16_K400.yaml \
  TRAIN.ENABLE False \
  TEST.CHECKPOINT_FILE_PATH path_to_your_checkpoint \
```

## Acknowledgement
This repository is built based on the [PySlowFast](https://github.com/facebookresearch/SlowFast).

## License
MeMViT is released under the [CC-BY-NC 4.0](LICENSE).
