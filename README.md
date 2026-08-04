# Layer-Gated Adapter (LGA)

## Overview

This repository contains the implementation of **Layer-Gated Adapter (LGA)**, a parameter-efficient fine-tuning framework for genomic foundation models.

---

## Repository Structure

```
.
├── run.py                 # Training and evaluation script
├── datasets/              # Example processed datasets
└── README.md
```

---

## Environment

The experiments were conducted with the following environment:

- Python 3.10
- PyTorch 2.1.0
- CUDA 12.1
- Transformers 4.57.3
- Datasets 2.18.0
- Biopython 1.86
- NumPy
- Scikit-learn

Please ensure that these dependencies are installed before running the code.

---

## Dataset

The `datasets/` directory contains **a subset of the processed datasets** used in our experiments for demonstration and code validation.

The complete benchmark consists of promoter datasets from 12 species constructed from publicly available databases, as described in our paper. The full processed dataset will be released after the publication of the paper.

The statistics of the benchmark dataset are summarized below:

| Species | Total samples | Positive samples | Training samples | Testing samples |
|:---|---:|---:|---:|---:|
| HoneyBee | 12,806 | 6,403 | 10,244 | 2,562 |
| Arabidopsis | 45,400 | 22,700 | 36,320 | 9,080 |
| Celegans | 14,240 | 7,120 | 11,392 | 2,848 |
| Dog | 13,930 | 6,965 | 11,144 | 2,786 |
| FruitFly | 33,942 | 16,971 | 27,153 | 6,789 |
| Zebrafish | 21,390 | 10,695 | 17,112 | 4,278 |
| Chicken | 12,248 | 6,124 | 9,798 | 2,450 |
| Human | 59,196 | 29,598 | 47,356 | 11,840 |
| Barleycorn | 42,386 | 21,193 | 33,908 | 8,478 |
| RhesusMacaque | 19,118 | 9,559 | 15,294 | 3,824 |
| Rat | 25,100 | 12,550 | 20,080 | 5,020 |
| BakerYeast | 10,226 | 5,113 | 8,180 | 2,046 |

Users may also prepare their own datasets following the same data format.

---

## Pre-trained Nucleotide Transformer

This project is built upon the **Nucleotide Transformer v2 (500M Multi-species)** pre-trained model.

Before running the code, please download the complete pre-trained model files from the official Hugging Face repository:

https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species

After downloading the model, modify the model path in `run.py` to point to your local checkpoint directory.

---

## Running

Training and evaluation are both implemented in `run.py`.

Run the experiment using:

```bash
python run.py
```

---
