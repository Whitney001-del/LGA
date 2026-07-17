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
- Accelerate 1.12.0
- Peft 0.18.0
- Tokenizers 0.22.1
- Flash_attn 2.8.3
- NumPy
- Scikit-learn

Please ensure that these dependencies are installed before running the code.

---

## Dataset

The `datasets/` directory contains **a subset of the processed datasets** used in our experiments for demonstration and code validation.

The complete benchmark consists of promoter datasets from 12 species constructed from publicly available databases, as described in our paper. The full processed dataset will be released after the publication of the paper.

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
