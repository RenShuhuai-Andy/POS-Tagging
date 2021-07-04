# Chinese-POS-tagging

POS-tagging for simplified and traditional Chinese data with BERT / RoBERTa. 

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

> sh pos_train.sh -b bert-wwm-ext -d simplified

- `-b` denotes the path to the pre-trained model.
- `-d` denotes the data type, which can be chosen from ['simplified', 'traditional'']
- `-c` denotes the index of CUDA device.

## Result

**Result on the dev set**

- **Simplified**

    |                              | loss           | accuracy       | precision       | recall          | f1               |
    | ---------------------------- | -------------- | -------------- | --------------- | --------------- | ---------------- |
    | BERT-base         | 0.18 |  95.79  |   95.24  |  95.63    | 95.44 |
    | roBERTa-large         | 0.21 |  96.17  |   95.92  |  96.09    | 96.00 |
    | roBERTa-large + FocalLoss         | 0.15 |  96.06  |   95.59  |  95.92    | 95.75 |
    | roBERTa-large + CRF         | 7.05 |  96.28  |   96.36  |  96.29    | 96.32 |

- **Traditional**

    |                              | loss           | accuracy       | precision       | recall          | f1               |
    | ---------------------------- | -------------- | -------------- | --------------- | --------------- | ---------------- |
    | roBERTa-large         | 0.25 |  94.66  |   94.20  |  94.55    | 94.38 |

**Result on the test set**

- **Simplified**

    |                              | loss           | accuracy       | precision       | recall          | f1               |
    | ---------------------------- | -------------- | -------------- | --------------- | --------------- | ---------------- |
    | BERT-base         | 0.18 |  95.84  |   95.29  |  95.71    | 95.50 |
    | roBERTa-large         | 0.20 |  96.32  |   96.05  |  96.23    | 96.14 |
    | roBERTa-large + FocalLoss         | 0.15 |  96.22  |   95.74  |  96.06    | 95.90 |
    | roBERTa-large + CRF         | 7.00 |  96.39  |   96.45  |  96.39    | 96.42 |
    
- **Traditional**

    |                              | loss           | accuracy       | precision       | recall          | f1               |
    | ---------------------------- | -------------- | -------------- | --------------- | --------------- | ---------------- |
    | roBERTa-large         | 0.24 |  94.62  | 94.21   |   94.52   | 94.36 |