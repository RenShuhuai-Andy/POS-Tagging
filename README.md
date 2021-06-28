# Chinese-POS-tagging

POS-tagging with BERT(RoBERTa). 

## Prepare environment
```bash
conda create -n pos python=3.6
conda activate pos
conda install pytorch torchvision cudatoolkit=10.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
pip install -r requirements.txt 
```
## Download pre-trained Chinese BERT

> sh download.sh bert-base

Change `bert-base` to `roberta-base` and `roberta-large` as you need.

## Train & eval

> sh pos_train.sh 

## Eval with official metric

> sh pos_eval.sh

## Result

**Result on the dev set**