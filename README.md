# Pytorch implementation for MulCPred
<img src="https://github.com/Equinoxxxxx/MulCPred_code/blob/master/fig1_00.png" width="500px">

## Customize the directories
Customize the directories for datasets (```dataset_root```) and weight files (```ckpt_root```) in config.py.

## Prepare the datasets
Download the datasets ([PIE](https://github.com/aras62/PIEPredict?tab=readme-ov-file#PIE_dataset),[TITAN](https://usa.honda-ri.com/titan))

Extract the data in the following structure:
```
[dataset_root]/
    PIE_dataset/
        annotations/
        PIE_clips/
        ...
    TITAN/
        honda_titan_dataset/
            dataset/
                clip_1/
                clip_2/
                ...
```
## Get the weights for pretrained backbones
Download the weight files to ```ckpt_root```
[C3D](https://drive.google.com/file/d/19NWziHWh1LgCcHU34geoKwYezAogv9fX/view?usp=sharing)
[HRNet](https://drive.google.com/open?id=1UoJhTtjHNByZSm96W3yFTfU5upJnsKiS)
```
[ckpt_root]/
    c3d-pretrained.pth
    pose_hrnet_w48_384x288.pth
```
Get cropped images and skeletons.
```
cd PedContrast
python -m tools.data.preprocess
```

## Train
# TITAN-crossing
```
python main.py --dataset_name TITAN
```

# TITAN-atomic
```
python main.py --dataset_name TITAN --use_atomic 1 --use_cross 0
```

# PIE
```
python main.py --dataset_name PIE
```

## Customize concepts
TBD
