# VRDL Final Project Swin Transformer with 8 classes

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

![](readme_img/single_swin_result.png)