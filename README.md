# Learn to Adapt for Generalized Zero-Shot Text Classiﬁcation

This repository is the official implementation of “Learn to Adapt for Generalized Zero-Shot Text Classiﬁcation”

## Requirements

To set our enviroment, we recommend to use `anaconda3` with GPU supports.

To install requirements:

```setup
pip install -r requirements.txt
```

## Datasets and Files

1. `data/` stores all raw datasets and `word2vec` files (except https://github.com/eyaler/word2vec-slim/blob/master/GoogleNews-vectors-negative300-SLIM.bin.gz，download it and put the unzip `bin` file on the directory) for BiLSTM. We use `huggingface transformer` package with `AutoTokenizer` and `AutoModel`

2. Firstly, you need to run preprocess scripts to store `pickle` files.

   ```shell
   python data_preprocess.py
   ```

   And `pickle` files will be stored in `data/{$data_name}`

## Training and Evaluation

We use configuration files to store hyperparameters and others in `config_{$data_name}.json`

For example, to train the **Metric Learning** model, run this command:

```shell
python train.py -d {$GPU_device} -st 1
```

where `-st` means step. We use step 1 for metric learning, and step 2 for LTA (w init ). If you want to run LTA w/o init,

you do not need to run step 1 first. And prototypes by metric learning `pickle` files are in `data/{$data_name}`

To train the LTA in the paper, run this command:

```shell
python train.py -d {$GPU_device} -st 2
```

Or you can change any configurations in the `json` file.

We use test set as our evaluation as other methods. We print the best perfomances with `logging` module.



For fairness, `train_run10.py` is for different random seeds and different seen/unseen class splits.

## Pre-trained Models

You can download pretrained models here:

We will open source code on our Github.

## Results

Our model achieves the following performance on :

### Generalized Zero-Shot Text Classification on CLINC

| Model name      | Accuracy HM | F1 HM     |
| --------------- | ----------- | --------- |
| Metric Learning | 52.61       | 58.06     |
| SOTA            | 16.31       | 10.18     |
| LTA             | **73.09**   | **73.47** |
