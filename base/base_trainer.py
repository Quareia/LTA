import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from utils import log_dict


class BaseTrainer:
    """Base class for all trainers."""
    def __init__(self, model, optimizer, config, dataset, epochs, save_per_epochs, verbosity, tensorboard, histogram, monitor, early_stop):
        self.config = config
        self.dataset = dataset
        self.logger = config.get_logger('train', verbosity)
        # self.valid_logger = config.get_logger('valid')

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.optimizer = optimizer

        # cfg_trainer = config['trainer']
        self.epochs = epochs
        self.save_per_epochs = save_per_epochs
        self.monitor = monitor

        self.histogram = histogram

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
            self.best_epoch = 0
            self.best_res = []
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = early_stop

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        # TODO: debug mode
        # if config.debug:
        tensorboard = False
        self.writer = TensorboardWriter(config.log_dir, self.logger, tensorboard)

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """Training logic for an epoch

        Args:
            epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """Full training logic."""
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.logger.info('\n'+'='*20)
            result = self._train_epoch(epoch)

            # save logged information into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged information to the screen
            log_dict(self.logger, log)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    self.best_epoch = epoch
                    not_improved_count = 0
                    self.best_res = [float('{:.2f}'.format(round(v * 100, 2))) for k, v in result.items() if 'val' in k]
                    if not self.config.debug:
                        self._save_checkpoint(epoch, save_best=True)
                else:
                    not_improved_count += 1
                    if not_improved_count >= int(self.early_stop):
                        break
                self.logger.info(
                    '*' * 20 + '    Best performance epoch {} : {:.6f}'.format(self.best_epoch, self.mnt_best))

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if not self.config.debug:
                if epoch % self.save_per_epochs == 0:
                    self._save_checkpoint(epoch)
                if self.histogram:
                    for name, p in self.model.named_parameters():
                        self.writer.add_histogram(name, p, bins='auto')
                if self.epochs == epoch:
                    self._save_checkpoint(self.epochs)

        best_path = str(self.checkpoint_dir / 'model_best.pth')
        self.logger.info('Best performance epoch {} : {:.6f} -- {}'.format(self.best_epoch,
                                                                           self.mnt_best,
                                                                           best_path))

        return self.best_res

    def _prepare_device(self, n_gpu_use):
        """setup GPU device if available, move model into configured device."""
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        """Saving checkpoints

        Args:
            epoch: current epoch number
            log: logging information of the epoch
            save_best: if True, save checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }

        if not save_best:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        else:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: {} ...".format(best_path))

    def _resume_checkpoint(self, resume_path):
        """Resume from saved checkpoints

        Args:
             resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
