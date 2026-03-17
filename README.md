# Simpsons Classification

This project contains a small pipeline for preparing a Simpsons image dataset and training a character classification model.

## Project Structure

```text
simpsons/
├── dedupe_simsons.py            # removes duplicates inside train/test and across splits
├── filter_simpsons_classes.py   # keeps only classes with enough training images
├── train_resnet18.py            # trains a ResNet18 classifier
├── simpsons_dataset/            # original training dataset
├── kaggle_simpson_testset/      # original test dataset
├── deduped_simpsons/            # dataset after deduplication
├── filtered_simpsons/           # final dataset after class filtering
├── filtered_simpsons.tar        # archived prepared dataset
└── output/                      # run artifacts
```

## Pipeline

1. `dedupe_simsons.py`
   - builds image embeddings with `ResNet50`
   - removes duplicates inside the training set
   - removes duplicates inside the test set
   - removes overlaps between train and test

2. `filter_simpsons_classes.py`
   - keeps only classes with enough training samples
   - creates final `filtered_simpsons/train` and `filtered_simpsons/test`

3. `train_resnet18.py`
   - trains `ResNet18` on `filtered_simpsons/train`
   - creates a validation split with `train_test_split`
   - saves metrics, confusion matrix, and the best checkpoint

## Requirements

Recommended environment: Python 3.10+

Main dependencies:

- `torch`
- `torchvision`
- `numpy`
- `pillow`
- `matplotlib`
- `scikit-learn`

Example installation:

```bash
pip install torch torchvision numpy pillow matplotlib scikit-learn
```

## Usage

Prepare the dataset:

```bash
python dedupe_simsons.py
python filter_simpsons_classes.py
```

Train the model:

```bash
python train_resnet18.py
```

## What To Commit

Good to keep in Git:

- source code
- `README.md`
- small example plots if needed

Better to exclude:

- `simpsons_dataset/`
- `kaggle_simpson_testset/`
- `deduped_simpsons/`
- `filtered_simpsons/`
- `filtered_simpsons.tar`
- `output/`
- `__pycache__/`
- `.idea/`

## First Push

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```
