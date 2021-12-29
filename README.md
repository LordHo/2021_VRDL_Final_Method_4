# VRDL Final Project Swin Transformer with 8 classes

## Environment

### Framework

Pytorch

### Installation

Download the whole project and install by following step.

```cmd
pip install -r requirement.txt
```

## Dataset

Dataset struct must be the following format.

- data
  - 8_class
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
- step1_classification
  - src

The val dataset is the first 10% data in each label, and other 90% is for train.  
You can download the whole same dataset by link.  
[dataset](https://drive.google.com/drive/folders/1wRcGU_abgKE5etkTI2OHfCvELABF9AC4?usp=sharing)

## Train

Excute the cell in `.\step1_classification\src\main.ipynb` like the following cell.

```python
from train import train
train()
```

## Eval

Excute the cell in `.\step1_classification\src\main.ipynb` like the following cell.

```python
from eval import eval
eval(stage=1)
```

If stage is 1(or 2), it'll ouput stage 1(or 2) image result as csv.

You should eval stage 1 and 2 to get two csv, then concate stage 2 result by following step.  
The merge can use   `.\step1_classification\src\merge_stage.ipynb`. Excute all cells.  
Be carefully fill the `stage1_csv_path`, `stage2_csv_path` and `merge_stage_csv_path` in  `.\step1_classification\src\merge_stage.ipynb`.  
The extra constant is default as 0.01, can modify as you wish.

[Merge sample](https://drive.google.com/file/d/1vgZPxopLcZRF9xqkFKvuxwgiMG6qBc2h/view?usp=sharing)  
[Add constant sample](https://drive.google.com/file/d/1FFL5rM5sCNg0BRGqzzXomfFgMLGJ5qm4/view?usp=sharing)  

## Result

| Model            | Private | Public  |
| ---------------- | ------- | ------- |
| Swin Transformer | 1.83721 | 1.18299 |

## Model Weights

[Swin Transform with 8 classes](https://drive.google.com/file/d/1G9kxAOVvf4vrhn3GlVX4jOJ5wMb5S_Jh/view?usp=sharing)
