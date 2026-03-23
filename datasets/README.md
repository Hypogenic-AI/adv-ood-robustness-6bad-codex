# Downloaded Datasets

This directory contains datasets for adversarial and OOD robustness experiments.
Large data files are excluded from git by `datasets/.gitignore`.

## Dataset 1: CIFAR-10

### Overview
- Source: Hugging Face `cifar10`
- Size: train 50,000, test 10,000
- Local size: ~131 MB on disk
- Format: Hugging Face Dataset saved to disk
- Task: 10-class image classification
- Splits: train, test
- License: see upstream CIFAR-10 terms

### Download Instructions

Using Hugging Face (recommended):
```python
from datasets import load_dataset

dataset = load_dataset("cifar10")
dataset.save_to_disk("datasets/cifar10")
```

### Loading the Dataset

```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/cifar10")
print(dataset)
```

### Sample Data

Saved at `datasets/cifar10/samples/samples.json` (5 examples).

### Notes
- Typical in-distribution benchmark for robustness work.
- Commonly paired with SVHN/TinyImageNet/CIFAR-10-C for OOD or corruption shifts.

## Dataset 2: SVHN (cropped_digits)

### Overview
- Source: Hugging Face `svhn` config `cropped_digits`
- Size: train 73,257, test 26,032, extra 531,131
- Local size: ~1.1 GB on disk
- Format: Hugging Face Dataset saved to disk
- Task: 10-class digit classification
- Splits: train, test, extra
- License: see upstream SVHN terms

### Download Instructions

Using Hugging Face (recommended):
```python
from datasets import load_dataset

dataset = load_dataset("svhn", "cropped_digits")
dataset.save_to_disk("datasets/svhn")
```

### Loading the Dataset

```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/svhn")
print(dataset)
```

### Sample Data

Saved at `datasets/svhn/samples/samples.json` (5 examples).

### Notes
- Useful as a near/far OOD dataset relative to CIFAR-10 depending on protocol.
- The `extra` split can be used for semi-supervised or calibration studies.

## Quick Validation Summary

Validation run completed during collection:
- Dataset loading and save-to-disk succeeded.
- Split counts confirmed (see `datasets/dataset_summary.json`).
- Sample records exported for both datasets.
