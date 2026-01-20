# VMBUNet
This is the official code repository for "VMB-UNet: A Boundary-Guided Cross-Level Feature Fusion Mamba-UNet for Dermoscopic Image Segmentation". 

## Abstract
Accurate segmentation of dermoscopic images is critical for the early diagnosis of skin cancer. However, existing methods still encounter substantial challenges, including insufficient modeling of global dependencies, progressive dilution of boundary information during cross-level feature fusion, and inadequate optimization of boundary pixels. These issues mainly stem from the low contrast, irregular morphology, and ambiguous boundaries commonly observed between lesion regions and normal skin. To address these challenges, we propose a boundary-guided cross-level feature fusion Mamba-UNet architecture, termed VMB-UNet, which enhances segmentation performance in ambiguous boundary regions through coordinated improvements in global context modeling, feature fusion, and optimization strategies. Specifically, VMB-UNet adopts Vision Mamba (VMamba) as the backbone and incorporates a State Space Context (SSC) module within a selective state space modeling framework. By integrating multi-scale contextual information, the SSC module strengthens the modeling of long-range dependencies and scale variations, providing more discriminative global features for boundary restoration. In addition, a Boundary Cross-Attention (BCA) module is introduced into shallow encoder layers and high-resolution decoder layers to explicitly enhance boundary awareness by generating boundary-focused attention maps. During decoding, a Boundary-guided Skip Fusion (BSF) module selectively refines skip-connected features under boundary guidance, effectively alleviating boundary information degradation during multi-scale fusion. Furthermore, a hybrid loss function combining a base segmentation loss with a distance-map-based boundary penalty is employed to impose stronger constraints on boundary pixels. Experiments conducted on the ISIC2017 and ISIC2018 datasets demonstrate that VMB-UNet achieves mIoU/Dice scores of 86.03\% / 92.49\% and 84.55\% / 91.55\%, respectively, consistently outperforming VM-UNet. These results validate the effectiveness and robustness of the proposed method for dermoscopic image segmentation with ambiguous boundaries.

## 0. Main Environments
```bash
conda create -n vmbunet python=3.10
conda activate vmbbunet
conda install cudatoolkit==11.8
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install setuptools==68.2.2
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```
The .whl files of causal_conv1d and mamba_ssm could be found here. {(https://github.com/Dao-AILab/causal-conv1d/releases) or (https://github.com/state-spaces/mamba/releases)}

## 1. Prepare the dataset

### ISIC datasets
- The ISIC17 and ISIC18 datasets can be found here {https://challenge.isic-archive.com/data/#2017 & https://challenge.isic-archive.com/data/#2018}. 

- After downloading the datasets, you are supposed to put them into './data/isic17/' and './data/isic18/', and the file format reference is as follows. 
- './data/isic17/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png

- './data/isic18/'
  - train
    - images
      - .jpg
    - masks
      - .png
  - val
    - images
      - .jpg
    - masks
      - .png

## 2. Prepare the pre_trained weights

- The weights of the pre-trained VMamba could be downloaded from [Baidu](https://pan.baidu.com/s/1ci_YvPPEiUT2bIIK5x8Igw?pwd=wnyy) or [GoogleDrive](https://drive.google.com/drive/folders/1ZJjc7sdyd-6KfI7c8R6rDN8bcTz3QkCx?usp=sharing). After that, the pre-trained weights should be stored in './pretrained_weights/'.

## 3. Train the VMBUNet
```bash
cd VMBUNet
python train.py  # Train and test VMBUNet on the ISIC17 or ISIC18 dataset.
```

## 4. Obtain the outputs
- After trianing, you could obtain the results in './results/'
