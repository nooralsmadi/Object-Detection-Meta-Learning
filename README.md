# Few-Shot Object Detection with Meta-Learning

This project implements and compares two object detection models in a **few-shot learning** scenario:

- A **baseline detector** trained with standard fine-tuning
- A **meta-learning-inspired variant** using a **cosine similarity head**

Both models are based on **Faster R-CNN** with a ResNet-50-FPN backbone and are evaluated on the **PASCAL VOC** 1-shot dataset (split1).  
The goal is to explore how meta-learning techniques improve performance when only **one labeled image per class** is available.

---

## Overview

- **Baseline model**: traditional linear classifier on ROI features.
- **Meta-learning model**: replaces the linear classifier with a **cosine similarity head**, inspired by metric-learning approaches like TFA (ICML 2020).
- **Dataset**: VOC 2007+2012 trainval for 1-shot training; VOC 2007 test for evaluation.
- **Metric**: Evaluation based on **mAP@50** over novel classes.

---

## Directory Structure

object-detection-meta-learning/
├── fsdet/ 
├── tools/ # Training and evaluation scripts
├── datasets/ # Few-shot data split generator
├── configs/ # YAML configs for both models
├── scripts/ # Shell scripts to automate the process 
├── metric/ # Evaluation results (.txt)
├── vis/ # Visual predictions
├── report.pdf # Final report
├── requirements.txt
└── README.md

---
## Install Dependencies

pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 \
  -f https://download.pytorch.org/whl/torch_stable.html

pip install detectron2==0.6+cu121 \
  -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.2/index.html

pip install -r requirements.txt

---
## Data Preparation

This project uses the **PASCAL VOC 2007 + 2012** dataset for few-shot object detection.

To prepare the dataset:


### Quick Setup (All-in-One Script)

To download PASCAL VOC and prepare the 1-shot few-shot splits (seed 1), simply run:

 scripts/download_voc.sh 


---
## Training & Evalauation 

python tools/train_net.py \
  --num-gpus 1 \
  --config-file configs/PascalVOC-detection/split1/ft_gpu_cosine.yaml


python tools/test_net.py \
  --num-gpus 1 \
  --config-file configs/PascalVOC-detection/split1/ft_gpu_baseline.yaml \
  --eval-only \
  --opts MODEL.WEIGHTS output/ft_gpu_baseline/model_final.pth



---
## Results

| Model       | Novel AP\@50 |
| ----------- | ------------ |
| Baseline    | 56.5%        |
| Cosine Head | 64.2%        |


---

## Visualizations

Sample outputs can be found in the vis/ directory:

base.jpg

meta_cosine.jpg

Each shows bounding boxes and predicted labels for the same test image.




