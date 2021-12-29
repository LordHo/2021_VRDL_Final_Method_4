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
You should eval stage 1 or 2 for both and concate stage 2 result following stage 1 result for update to kaggle.

## Result

| Model            | Private | Public  |
| ---------------- | ------- | ------- |
| Swin Transformer | 1.83721 | 1.18299 |

## Model Weights

[Swin Transform with 8 classes](https://drive.google.com/file/d/1G9kxAOVvf4vrhn3GlVX4jOJ5wMb5S_Jh/view?usp=sharing)
