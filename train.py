import argparse
import collections

from data_loader.dataset import Dataset
import data_loader.data_sampler as module_sampler
import model.loss as module_loss
import model.network.model as module_arch
from parse_config import ConfigParser
from trainer import trainer as module_trainer
from data_preprocess import *
from utils import write_json
from collections import defaultdict

# fix random seeds for reproducibility
SEED = 18
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(config):
    # Logger
    # logger = config.get_logger('train')

    # Data
    dataset = Dataset(config['dataset']['load_path'])

    # Config
    # # save updated config file to the checkpoint dir
    encoder_type = config['encoder_type']
    if encoder_type != 'bert':  # lstm needs word2vec
        config.deep_update_config(config.config, dataset.corpus.config())
    if not config.debug:
        write_json(config.config, config.save_dir / 'config_{}.json'.format(config['dataset']['name']))

    # Train
    # # Loss function
    loss_fn = getattr(module_loss, config['loss'])

    if config['step1']:
        # Step 1: pre-training using metric-learning for initialization
        if encoder_type == 'bert':
            model = config.init_obj('arch_step1', module_arch, config=config, encoder_type=encoder_type)

        else:
            model = config.init_obj('arch_step1', module_arch, config=config, encoder_type=encoder_type, corpus=dataset.corpus)

        trainable_params = [
            {"params": [p for n, p in model.named_parameters()], 'lr': config['arch_step1']['lr']}
        ]
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        trainer = config.init_obj('trainer_step1',
                                  module_trainer,
                                  model=model,
                                  optimizer=optimizer,
                                  config=config,
                                  dataset=dataset,
                                  loss_fn=loss_fn,
                                  lr_scheduler=lr_scheduler)
        trainer.train()

    if config['step2']:
        # Step 2: meta-learning procedure

        # # Sampler
        train_y = [sample['y'] for sample in dataset.train_seen]
        data_sampler = config.init_obj('data_sampler', module_sampler, labels=train_y)

        if encoder_type == 'bert':
            model = config.init_obj('arch_step2', module_arch, config=config, encoder_type=encoder_type, n_seen_class=dataset.n_seen_class)

        else:
            model = config.init_obj('arch_step2', module_arch, config=config, encoder_type=encoder_type, n_seen_class=dataset.n_seen_class, corpus=dataset.corpus)

        # # Initialization using step1 output
        if config['arch_step2']['ablation']['init']:
            with torch.no_grad():
                with open('./data/ver1/{}/protos_{}.pkl'.format(config['dataset']['name'],
                                                        config['encoder_type']), 'rb') as f:
                    protos = pickle.load(f)
                for i in range(len(model.seen_class_protos)):
                    model.seen_class_protos[i] = torch.nn.Parameter(protos[i])

        # # set different learning rates for the different parts of the model
        param_lr_dict = {param_name: lr for lr, param_name_list in config['arch_step2']['lr'].items() for param_name in param_name_list}
        trainable_params = defaultdict(list)
        for name, param in model.named_parameters():
            flag = False
            for p, lr in param_lr_dict.items():
                if name.find(p) != -1:
                    flag = True
                    trainable_params[float(lr)].append(param)
            if not flag:
                trainable_params[float(param_lr_dict['others'])].append(param)
        trainable_params = list(map(lambda x, y: {**x, **y}, [{'lr': lr} for lr in trainable_params.keys()],
                                    [{'params': param} for param in trainable_params.values()]))

        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        trainer = config.init_obj('trainer_step2',
                                  module_trainer,
                                  model=model,
                                  optimizer=optimizer,
                                  config=config,
                                  dataset=dataset,
                                  loss_fn=loss_fn,
                                  data_sampler=data_sampler,
                                  lr_scheduler=lr_scheduler)
        trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='GZSL Research')
    # args.add_argument('-c', '--config', default='config_SNIPS.json', type=str,
    # args.add_argument('-c', '--config', default='config_SMP.json', type=str,
    # args.add_argument('-c', '--config', default='config_ATIS.json', type=str,
    args.add_argument('-c', '--config', default='config_Clinc_LSTM.json', type=str,
                      # args.add_argument('-c', '--config', default='config_Quora.json', type=str,
                      # args.add_argument('-c', '--config', default='config_Samsung.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='2', type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-dg', '--debug', default=True, type=bool,
                      help='debug mode')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-st1', '--step1'], type=bool, target='step 1'),
        CustomArgs(['-st2', '--step2'], type=bool, target='step 2')
    ]

    config = ConfigParser.from_args(args, options)
    main(config)
