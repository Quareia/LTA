# Learn to Adapt for Generalized Zero-Shot Text Classification

This repository is the official implementation of ["Learn to Adapt for Generalized Zero-Shot Text Classification"](https://aclanthology.org/2022.acl-long.39.pdf) (ACL 2022 main conference).
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
```
@inproceedings{zhang-etal-2022-learn,
    title = "Learn to Adapt for Generalized Zero-Shot Text Classification",
    author = "Zhang, Yiwen  and
      Yuan, Caixia  and
      Wang, Xiaojie  and
      Bai, Ziwei  and
      Liu, Yongbin",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.39",
    doi = "10.18653/v1/2022.acl-long.39",
    pages = "517--527",
    abstract = "Generalized zero-shot text classification aims to classify textual instances from both previously seen classes and incrementally emerging unseen classes. Most existing methods generalize poorly since the learned parameters are only optimal for seen classes rather than for both classes, and the parameters keep stationary in predicting procedures. To address these challenges, we propose a novel Learn to Adapt (LTA) network using a variant meta-learning framework. Specifically, LTA trains an adaptive classifier by using both seen and virtual unseen classes to simulate a generalized zero-shot learning (GZSL) scenario in accordance with the test time, and simultaneously learns to calibrate the class prototypes and sample representations to make the learned parameters adaptive to incoming unseen classes. We claim that the proposed model is capable of representing all prototypes and samples from both classes to a more consistent distribution in a global space. Extensive experiments on five text classification datasets show that our model outperforms several competitive previous approaches by large margins. The code and the whole datasets are available at https://github.com/Quareia/LTA.",
}
```

# Other Configurations
I am sorry that the configurations of BERT and other datasets are lost. I don't have time at the moment to rerun the experiments, but I will do it sooner or later (I have no machine because I have already graduated..). But I do think a non-pretrained encoder will be better for GZSL research, I think it more like a ML question rather than a NLP question.
