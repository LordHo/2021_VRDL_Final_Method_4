# VRDL Final Project Swin Transformer with 8 classes

## Dataset

Dataset struct must be the following format.

- data
  - train
    - ALB
      - sample.jpg
    - BET
    - DOL
    - LAG
    - NoF
    - OTHER
    - SHARK
    - YFT
  - val (with same struct as train)
  - test_stg1
    - sample.jpg
  - test_stg2
    - sample.jpg

The val dataset is the first 10% data in each label, and other 90% is for train.

## Train

Excute the cell in main.ipynb like the following cell.

```python
from train import train
train()
```

## Eval

Excute the cell in main.ipynb like the following cell.

```python
from eval import eval
eval(stage=1)
```

If stage is 1(2), it'll ouput stage 1(2) image result as csv.

## Result

| Model            | Private | Public  |
| ---------------- | ------- | ------- |
| Swin Transformer | 1.83721 | 1.18299 |
