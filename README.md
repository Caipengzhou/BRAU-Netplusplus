# BRAU-Net++: U-Shaped Hybrid CNN-Transformer Network for Medical Image Segmentation
- Accurate medical image segmentation is essential for clinical quantification, disease diagnosis, treatment planning and many other applications. Both convolution-based and transformer-based u-shaped architectures have made 
significant success in various medical image segmentation tasks. The former can efficiently learn local information of images while requiring much more image-specific inductive biases inherent to convolution operation. 
The latter can effectively capture longrange dependency at different feature scales using selfattention, whereas it typically encounters the challenges of quadratic compute and memory requirements with sequence length increasing. 
To address this problem, through integrating the merits of these two paradigms in a welldesigned u-shaped architecture, we propose a hybrid yet effective CNN-Transformer network, named BRAU-Net++,
for an accurate medical image segmentation task. Specifically, BRAU-Net++ uses bi-level routing attention as the core building block to design our u-shaped encoderdecoder structure, in which both encoder and decoder are
hierarchically constructed, so as to learn global semantic information while reducing computational complexity. Furthermore, this network restructures skip connection by incorporating channel-spatial attention which adopts
convolution operations, aiming to minimize local spatial information loss and amplify global dimension-interaction of multi-scale features. Extensive experiments on three public benchmark datasets demonstrate that our proposed
approach surpasses other state-of-the-art methods including its baseline: BRAU-Net under almost all evaluation metrics. We achieve the average Dice-Similarity Coefficient(DSC) of 82.47, 90.10, and 92.94 on Synapse multi-organ
segmentation, ISIC-2018 Challenge, and CVC-ClinicDB, as well as the mIoU of 84.01 and 88.17 on ISIC-2018 Challengeand CVC-ClinicDB, respectively. 

## 1. Synapse data and Model weights
- Get Synapse data, model weights of our BRAU-Net++ on the Synapse dataset and biformer_base_best.pth. (https://drive.google.com/file/d/115-vkjCapans_Mx3EXLxZsxr_WSbpXxm/view?usp=sharing). I hope this will help you to reproduce the results.
## 2. Environment
- Please prepare an environment with python=3.10, and then use the command "pip install -r requirements.txt" for the dependencies.
## 3. Synapse Train/Test
- Train
```bash
python train.py --dataset Syanpse --root_path your DATA_DIR --max_epochs 400 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.05 --batch_size 24
```
- Test 

```bash
python test.py --dataset Synapse --is_savenii --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 400 --base_lr 0.05 --img_size 224 --batch_size 24
```

## References
* [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
* [BiFormer](https://github.com/rayleizhu/BiFormer)
* [DAEFormer](https://github.com/xmindflow/DAEFormer)
* [DCSAU-Net](https://github.com/xq141839/DCSAU-Net)

