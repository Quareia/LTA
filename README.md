# Learn to Adapt for Generalized Zero-Shot Text Classification

This repository is the official implementation of “Learn to Adapt for Generalized Zero-Shot Text Classification” (ACL 2022 main conference).
The structure of our repository is based on the template https://github.com/victoresque/pytorch-template.

## Requirements

To set our environment, we recommend to use `anaconda3` with GPU supports.

```shell
conda create -n lta python=3.7.9
conda activate lta
```

After we setup basic conda environment, install pytorch `torch=1.7.1` and install requirements:

```shell
pip install -r requirements.txt
```

## Datasets and Files

1. For `BiLSTM` encoder, download `word2vec` English resource https://github.com/eyaler/word2vec-slim/blob/master/GoogleNews-vectors-negative300-SLIM.bin.gz, 
and put the unzip `bin` file to the directory `data/resources`. For `BERT` encoder, we use `huggingface transformer` package with `AutoTokenizer` and `AutoModel`.

2. Preprocess data files to `pickle` binary file. Run preprocess scripts:

   ```shell
   python data_preprocess.py
   ```

   And `pickle` files will be stored in `data/ver{$version}/{$data_name}`. We give the splitting version mentioned in 
   our paper, and you can split and keep the raw `.csv` files by yourself to other `$version`.

## Training and Evaluation

We use configuration files to store hyper-parameters for experiments in `config_{$data_name}_{$encoder_type}.json`

For example, to train the **Metric Learning** model, run this command:

```shell
python train.py -d {$GPU_device} -st1 1
```

where `-st1` means step 1. If you want to run LTA w/o init, you do not need to run step 1 first. 
All prototypes `pickle` files will be stored to `data/ver{$version}/{$data_name}/protos_{$encoder_type}.pkl`.

To train the LTA in the paper, run step 2:

```shell
python train.py -d {$GPU_device} -st2 1
```

Or you can change any configurations in the `json` file.

Debug mode is also provided, which does not generate experiment directory.
 
# Citation