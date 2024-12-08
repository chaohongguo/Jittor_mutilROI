# Multi-RoI Human Mesh Recovery with Camera Consistency and Contrastive Losses

**Jittor version** of Code repository for the paper:

Multi-RoI Human Mesh Recovery with Camera Consistency and Contrastive Losses[ECCV2024]

[paper](https://arxiv.org/abs/2402.02074) [project](https://github.com/CptDiaos/Multi-RoI?tab=readme-ov-file)

## Installation instructions

```
conda create -n mutilROI pytyon=3.9
conda activate mutilROI
pip install jittor
python -m jittor.test.test_example
pip install numpy==1.23.5
pip install -r requirements.txt
```

## Fetch data

let your data path as following

```
.data
├── best_ckpt
|   ├── jittor_best.pkl 
├── dataset_extras
│   ├── coco_train.npz
│   ├── 3dpw_test_w2d_smpl3d_gender.npz
│   ├── 3dpw_train_w2d_smpl3d_gender.npz
|   └── ......npz
├── smpl
│   ├── SMPL_FEMALE.pkl
│   ├── smpl_kid_template.npy
│   ├── SMPL_MALE.pkl
│   └── SMPL_NEUTRAL.pkl
├── J_regressor_extra.npy
├── J_regressor_h36m.npy
└── smpl_mean_params.npz
```

download part of dataset_extras from [here]()

download smpl from [here](https://www.bing.com/search?q=smpl&cvid=57cea722454a4e58a2ec22a8fd748ba3&gs_lcrp=EgRlZGdlKgYIABBFGDsyBggAEEUYOzIGCAEQABhAMgYIAhBFGDsyBggDEAAYQDIGCAQQABhAMgYIBRBFGDwyBggGEEUYPDIGCAcQRRg8MgYICBBFGEHSAQgxMjg3ajBqOagCCLACAQ&FORM=ANAB01&adppc=EdgeStart&PC=LCTS)

## Eval

```
python eval.py --model_name mutilROI --checkpoint data/best_ckpt/jittor_best.pkl
```

## Train

```
>>> train on single datset
python train.py --fixname --name example_train
--lr 1e-4 --model_name mutilROI
--viz_debug --train_dataset coco
--bbox_type rect
--batch_size 36
--use_aug_trans


>>> train on mixed datasets
python train.py --fixname --name example_train
--lr 1e-4 --model_name mutilROI
--viz_debug
--bbox_type rect
--batch_size 36
--use_aug_trans
```

