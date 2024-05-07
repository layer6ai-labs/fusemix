<div align="center">
<h1>
<b>
Data-Efficient Multimodal Fusion on a Single GPU
</b>
</h1>

<p align="center">
  <a href='https://arxiv.org/abs/2312.10144'><img src='https://img.shields.io/badge/arXiv-2312.10144-b31b1b.svg' /></a>
</p>
  
<h4>
<b>
<a href="https://www.cs.toronto.edu/~nvouitsis/">Noël Vouitsis*</a>, <a href="https://www.linkedin.com/in/zhaoyan-liu-9309aa180/">Zhaoyan Liu*</a>, <a href="https://www.cs.toronto.edu/~satyag/">Satya Krishna Gorti*</a>, <a href="http://linkedin.com/in/valentin-villecroze">Valentin Villecroze</a>, <a href="http://jescresswell.github.io/">Jesse C. Cresswell</a>, <a href="http://www.cs.toronto.edu/~guangweiyu/">Guangwei Yu</a>, <a href="https://sites.google.com/view/gabriel-loaiza-ganem/">Gabriel Loaiza-Ganem</a>, <a href="https://www.cs.toronto.edu/~mvolkovs/">Maksims Volkovs</a>    
</b>
</h4>
</div>


## Introduction
This repository contains the official implementation of our <b>CVPR 2024 Highlight</b> paper <a href='https://arxiv.org/abs/2312.10144'>Data-Efficient Multimodal Fusion on a Single GPU</a>. We release code for the image-text setting, including code for dataset downloading, feature extraction, fusion training and evaluation. We note that our code is based on the [LAVIS](https://github.com/salesforce/LAVIS) library.

## Installation

1. (Optional) Creating conda environment

```bash
conda create -n fusemix python=3.8
conda activate fusemix
```
 
2. Build from source

```bash
git clone https://github.com/layer6ai-labs/fusemix
cd fusemix
pip install -e .
```

## Getting Started
### Model Zoo
Model zoo summarizes supported models, to view:
```python
from lavis.models import model_zoo
print(model_zoo)
# ======================================================================
# Architectures                            Types
# ======================================================================
# dinov2_feature_extractor                 vits14, vitb14, vitl14, vitg14
# bge_feature_extractor                    large
# cohere_feature_extractor                 v3
# mlp_contrastive_fusion                   base
```

### Dataset Zoo
Dataset zoo summarizes supported datasets, to view:

```python
from lavis.datasets.builders import dataset_zoo
dataset_names = dataset_zoo.get_names()
print(dataset_names)
```

### Dataset Downloading
Please refer to `lavis/datasets/download_scripts` for scripts to download the required datasets.


### Feature Extraction

```bash
bash run_scripts/feature_extract/feat_extract_bge_large_coco_cap.sh
```


### FuseMix Training

```bash
bash run_scripts/fusion/mlp_contrastive_fusion_pretrain_dinov2_vitg14_bge_large_coco_vg_sbu_cap_cc3m.sh
```

### Evaluation

```bash
bash run_scripts/fusion/mlp_contrastive_fusion_retrieval_dinov2_vitg14_bge_large_coco.sh
```

## Citation
If you find this work useful in your research, please cite the following paper:
```
@inproceedings{vouitsis2024dataefficient,
      title={Data-Efficient Multimodal Fusion on a Single GPU}, 
      author={No{\"e}l Vouitsis and Zhaoyan Liu and Satya Krishna Gorti and Valentin Villecroze and Jesse C. Cresswell and Guangwei Yu and Gabriel Loaiza-Ganem and Maksims Volkovs},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2024},
}
```
