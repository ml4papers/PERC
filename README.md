# PERC

Official code release for the paper  
**PERC: Perception-Driven Calibration for Balanced Multimodal Test-Time Adaptation**


## Benchmarks

We conduct experiments on the **VGGSound** and **Kinetics** datasets.
Corruptions for both video and audio modalities are generated following the
implementation provided in the **READ** repository
([GitHub](https://github.com/XLearning-SCU/2024-ICLR-READ)).
All datasets should be placed in the `data_path` directory under the project root.

Pretrained models for VGGSound and KS50 are available in the READ repository
([GitHub](https://github.com/XLearning-SCU/2024-ICLR-READ)).



## Reproduce
python run_read.py --dataset 'ks50' --json-root 'code_path/json_csv_files/ks50' --label-csv 'code_path/json_csv_files/class_labels_indices_ks50.csv' --pretrain_path 'code_path/pretrained_model/cav_mae_ks50.pth' --tta-method 'PERC' --severity-start 5 --severity-end 5 --corruption-modality 'video'

## Enviroments
python== 3.9
torch==1.13.1
torchaudio==0.13.1
timm==0.6.5
scikit-learn==0.22.1
numpy==1.21.6

## Acknowledgement

This repository is based on and derived from the implementation accompanying
the paper **ABPEM** (https://github.com/YushengZhao/ABPEM), **READ** (https://github.com/XLearning-SCU/2024-ICLR-READ) and **2LTTA** (https://github.com/ml4papers/2LTTA). The original code is licensed under the Apache License 2.0.
