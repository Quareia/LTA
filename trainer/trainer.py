import numpy as np
import torch
from torch.utils.data import BatchSampler, RandomSampler
from data_loader.data_sampler import CLSBatchSampler
from base import BaseTrainer
from utils import MetricTracker, get_samples, log_dict
from model.loss import *
from model.metric import *
import pickle
from numpy import inf


class StepOneTrainer(BaseTrainer):
    """Metric-Learning Supervised Classification Trainer Class"""

    def __init__(self, model, optimizer, config, dataset, loss_fn,
                 epochs,
                 len_epoch,
                 save_per_epochs, verbosity,
                 train_batch_size,
                 valid_batch_size,
                 tensorboard, histogram, monitor='off', early_stop=inf, lr_scheduler=None):
        super().__init__(model, optimizer, config, dataset, epochs, save_per_epochs, verbosity, tensorboard, histogram, monitor, early_stop)
        self.loss_fn = loss_fn
        self.data_sampler = CLSBatchSampler(range(len(self.dataset.train_seen)),
                                    batch_size=train_batch_size)
        self.len_epoch = len_epoch

        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(len(self.data_sampler)))

        self.train_metrics = MetricTracker('loss',
                                           'seen_accuracy',
                                           'seen_precision',
                                           'seen_recall',
                                           'seen_macro_f1',
                                           writer=self.writer)
        self.valid_batch_size = valid_batch_size

    def _train_epoch(self, epoch):
        """Training logic for an epoch

        Args:
            epoch: Integer, current training epoch.

        Returns:
            A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, idxs in enumerate(self.data_sampler):
            if batch_idx >= self.len_epoch:
                break
            class_samples = self.dataset.seen_class
            class_protos_x, class_protos_len, class_protos_y = get_samples(class_samples, self.device, self.config['encoder_type'])

            query_samples = [self.dataset.train_seen[ids] for ids in idxs]
            querys_x, querys_len, querys_y = get_samples(query_samples, self.device, self.config['encoder_type'])

            self.optimizer.zero_grad()

            protos = self.model(class_protos_x, class_protos_len, 'encode')
            querys = self.model(querys_x, querys_len, 'encode')
            loss, output = self.loss_fn(protos, querys, querys_y, self.model.tau)

            # L2 regularization
            # L2_reg = torch.tensor(0., requires_grad=True).to(self.device)
            # for name, param in self.model.named_parameters():
            #     if 'lstm' in name:
            #         L2_reg = L2_reg + param.norm(p=2)
            # lambda_reg = 1e-5
            # loss_reg = lambda_reg * L2_reg

            loss_total = loss# + loss_reg

            loss_total.backward()
            self.optimizer.step()

            # train_metrics
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            # # loss
            self.train_metrics.update('loss', loss.item())
            # # metrics
            y = querys_y.cpu()
            y_pred = output.max(dim=1)[1].cpu()
            self.train_metrics.update('seen_accuracy', accuracy_fn(y, y_pred))
            self.train_metrics.update('seen_precision', precision_fn(y, y_pred))
            self.train_metrics.update('seen_recall', recall_fn(y, y_pred))
            self.train_metrics.update('seen_macro_f1', macro_f1_fn(y, y_pred))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} Batch id: {} Loss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    loss.item()))
        log = self.train_metrics.result()

        # validation
        self.logger.info('=' * 5 + ' valid')
        val_log = self._valid_epoch(epoch)
        log.update(**{'val_' + k: v for k, v in val_log.items()})

        # print logged information to the logging
        log_dict(self.logger, val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.config['loss'] == 'CosT':
            self.logger.info('tau : {}'.format(self.model.tau))

        return log

    @torch.no_grad()
    def _valid_epoch(self, epoch):
        """Validate after training an epoch

        Args:
            epoch: Integer, current training epoch.

        Returns:
            A log that contains information about validation
        """
        self.model.eval()

        # n_seen_class = self.dataset.n_seen_class
        start = - int(self.dataset.unseen_class[0]['y']) + self.dataset.n_seen_class

        # proto
        class_samples = self.dataset.seen_class + self.dataset.unseen_class
        class_protos_x, class_protos_len, class_protos_y = get_samples(class_samples, self.device, self.config['encoder_type'])

        protos = self.model(class_protos_x, class_protos_len, 'encode')

        # y_true and logit_pred for calculating all metric.
        ys = torch.LongTensor([])
        ps = torch.tensor([])

        # val seen
        n_val = len(self.dataset.test_seen)
        valid_data_sampler = CLSBatchSampler(range(n_val), self.valid_batch_size)

        for batch_idx, idxs in enumerate(valid_data_sampler):
            query_samples = [self.dataset.test_seen[ids] for ids in idxs]
            querys_x, querys_len, querys_y = get_samples(query_samples, self.device, self.config['encoder_type'])

            querys = self.model(querys_x, querys_len, 'encode')

            loss, output = self.loss_fn(protos, querys, querys_y)

            # valid_metrics
            self.writer.set_step((epoch - 1) * len(valid_data_sampler) + batch_idx, 'valid')
            # # metrics
            # y = querys_y.cpu()
            # y_pred = output.max(dim=1)[1].cpu()
            ys = torch.cat([ys, querys_y.cpu()], 0)
            ps = torch.cat([ps, output.cpu()], 0)

        # val unseen
        n_val = len(self.dataset.test_unseen)
        valid_data_sampler = CLSBatchSampler(range(n_val), self.valid_batch_size)

        for batch_idx, idxs in enumerate(valid_data_sampler):
            query_samples = [self.dataset.test_unseen[ids] for ids in idxs]
            querys_x, querys_len, querys_y = get_samples(query_samples, self.device, self.config['encoder_type'])
            querys_y = querys_y + start  # Reset testing y label to the right start label.

            querys = self.model(querys_x, querys_len, 'encode')

            loss, output = self.loss_fn(protos, querys, querys_y)

            # valid_metrics
            self.writer.set_step((epoch - 1) * len(valid_data_sampler) + batch_idx, 'valid')
            # y = querys_y.cpu()
            ys = torch.cat([ys, querys_y.cpu()], 0)
            ps = torch.cat([ps, output.cpu()], 0)

        res, _ = all_metric(ys,
                            ps,
                            len(self.dataset.test_seen),
                            self.dataset.n_seen_class,
                            self.dataset.n_unseen_class)
        # self.logger.info('\t'.join(['{:.2f}'.format(round(_ * 100, 2)) for _ in res.values()]))

        #TODO
        if self.mnt_mode != 'off' and res['GZSL_acc_hm'] >= self.mnt_best:
            with open('./data/ver1/{}/protos_{}.pkl'.format(self.config['dataset']['name'], self.config['encoder_type']), 'wb') as f:
                pickle.dump(protos.detach().cpu(), f)
                print('Save proto pickle file at {} epoch'.format(epoch))
        return res


class StepTwoTrainer(BaseTrainer):
    """Generalized Meta-Learning Trainer Using LTA framework."""

    def __init__(self, model, optimizer, config, dataset, loss_fn, data_sampler,
                 epochs, save_per_epochs, verbosity,
                 valid_batch_size,
                 tensorboard, histogram, monitor='off', early_stop=inf, lr_scheduler=None):
        super().__init__(model, optimizer, config, dataset, epochs, save_per_epochs, verbosity, tensorboard, histogram,
                         monitor, early_stop)
        self.loss_fn = loss_fn
        self.data_sampler = data_sampler
        self.len_epoch = len(self.data_sampler)

        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(len(data_sampler)))

        self.train_metrics = MetricTracker('loss',
                                           'seen_accuracy',
                                           'seen_precision',
                                           'seen_recall',
                                           'seen_macro_f1',
                                           'unseen_accuracy',
                                           'unseen_precision',
                                           'unseen_recall',
                                           'unseen_macro_f1',
                                           'GZSL_seen_recall',
                                           'GZSL_unseen_recall',
                                           'GZSL_acc_hm',
                                           'GZSL_f1_hm',
                                           writer=self.writer)
        self.valid_batch_size = valid_batch_size

    def _train_epoch(self, epoch):
        """Training logic for an epoch

        Args:
            epoch: Integer, current training epoch.

        Returns:
            A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (unseen_classes, unseen_query, seen_classes, seen_query) in enumerate(self.data_sampler):
            self.optimizer.zero_grad()

            n_seen = len(seen_classes)
            unseen_class_samples = [self.dataset.seen_class[idx] for idx in unseen_classes]
            unseen_protos_x, unseen_protos_len, unseen_protos_y = get_samples(unseen_class_samples, self.device, self.config['encoder_type'])
            unseen_protos = self.model(unseen_protos_x, unseen_protos_len, 'encode')

            unseen_class_idxs = torch.tensor(unseen_classes)
            seen_class_idxs = torch.tensor(seen_classes)

            # seen_class_idxs = (~(torch.arange(self.dataset.n_seen_class)[..., None] == unseen_class_idxs).any(
            #     -1)).nonzero(as_tuple=False).squeeze(-1)
            # unseen_protos = protos[unseen_class_idxs]

            protos, semantic_components = self.model(unseen_protos, seen_class_idxs, 'proto_adapt')

            # protos, memory_protos, novel_protos, after_memory_protos, after_novel_protos, v, loss_r = \
                # self.model(novel_protos, memory_class_idxs, novel_class_idxs,'transfer')

            unseen_query_samples = [self.dataset.train_seen[ids] for ids in unseen_query]
            unseen_querys_x, unseen_querys_len, unseen_querys_y = get_samples(unseen_query_samples, self.device, self.config['encoder_type'])
            seen_query_samples = [self.dataset.train_seen[ids] for ids in seen_query]
            seen_querys_x, seen_querys_len, seen_querys_y = get_samples(seen_query_samples, self.device, self.config['encoder_type'])

            re_index = torch.sort(torch.cat([seen_class_idxs, unseen_class_idxs], 0).to(self.device))[1]
            unseen_querys_y = re_index[unseen_querys_y]
            seen_querys_y = re_index[seen_querys_y]

            # if epoch >= 15:
            # novel_querys = self.model(novel_querys_x, novel_querys_len, memory_protos, novel_protos,
            #                                      after_memory_protos, after_novel_protos, v, 'adapt')
            # memory_querys = self.model(memory_querys_x, memory_querys_len, memory_protos, novel_protos,
            #                                       after_memory_protos, after_novel_protos, v, 'adapt')

            unseen_querys = self.model(unseen_querys_x, unseen_querys_len, protos, semantic_components, n_seen,
                                       'sample_adapt')
            seen_querys = self.model(seen_querys_x, seen_querys_len, protos, semantic_components, n_seen, 'sample_adapt')


            loss_unseen, output_unseen = self.loss_fn(protos, unseen_querys, unseen_querys_y, self.model.tau)
            loss_seen, output_seen = self.loss_fn(protos, seen_querys, seen_querys_y, self.model.tau)

            # loss_n, output_n = self.loss_fn(protos, novel_querys, novel_querys_y, self.model.tao_cos)
            # loss_m, output_m = self.loss_fn(protos, memory_querys, memory_querys_y, self.model.tao_cos)

            # L2 regularization
            # L2_reg = torch.tensor(0., requires_grad=True).to(self.device)
            # for name, param in self.model.named_parameters():
            #     if ('lstm' in name) or ('attn' in name) or ('generator' in name):
            #         L2_reg = L2_reg + param.norm(p=2)
            # lambda_reg = 1e-5
            # loss_reg = lambda_reg * L2_reg

            loss = loss_unseen + loss_seen
            loss_total = loss# + loss_reg

            loss_total.backward()
            self.optimizer.step()

            # train_metrics
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            y = seen_querys_y.cpu()
            y_pred = output_seen.max(dim=1)[1].cpu()

            self.train_metrics.update('seen_accuracy', accuracy_fn(y, y_pred))
            self.train_metrics.update('seen_precision', precision_fn(y, y_pred))
            self.train_metrics.update('seen_recall', recall_fn(y, y_pred))
            self.train_metrics.update('seen_macro_f1', macro_f1_fn(y, y_pred))
            self.train_metrics.update('GZSL_seen_recall', recall_fn(torch.ones_like(y),
                                                                     y_pred < n_seen,
                                                                     average='binary'))

            y = unseen_querys_y.cpu()
            y_pred = output_unseen.max(dim=1)[1].cpu()
            self.train_metrics.update('unseen_accuracy', accuracy_fn(y, y_pred))
            self.train_metrics.update('unseen_precision', precision_fn(y, y_pred))
            self.train_metrics.update('unseen_recall', recall_fn(y, y_pred))
            self.train_metrics.update('unseen_macro_f1', macro_f1_fn(y, y_pred))
            self.train_metrics.update('GZSL_unseen_recall', recall_fn(torch.ones_like(y),
                                                                     y_pred >= n_seen,
                                                                     average='binary'))
            self.train_metrics.update('GZSL_acc_hm', HM_fn(self.train_metrics.avg('seen_accuracy'),
                                                            self.train_metrics.avg('unseen_accuracy')))
            self.train_metrics.update('GZSL_f1_hm', HM_fn(self.train_metrics.avg('seen_macro_f1'),
                                                            self.train_metrics.avg('unseen_macro_f1')))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} Batch id: {} Loss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    loss.item()))

        log = self.train_metrics.result()

        # validation
        self.logger.info('=' * 5 + ' valid')
        val_log = self._valid_epoch(epoch)
        log.update(**{'val_' + k: v for k, v in val_log.items()})

        # print logged information to the logging
        log_dict(self.logger, val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.config['loss'] == 'CosT':
            self.logger.info('tau : {}'.format(self.model.tau))

        return log

    @torch.no_grad()
    def _valid_epoch(self, epoch):
        """Validate after training an epoch

        Args:
            epoch: Integer, current training epoch.

        Returns:
            A log that contains information about validation
        """
        self.model.eval()

        n_seen = self.dataset.n_seen_class
        start = - int(self.dataset.unseen_class[0]['y']) + self.dataset.n_seen_class

        ys = torch.LongTensor([])
        ps = torch.tensor([])

        seen_class_idxs = torch.tensor(range(n_seen))
        # proto

        # memory_query_samples = self.dataset.seen_class
        # memory_protos_x, memory_protos_len, memory_protos_y = get_samples(memory_query_samples, self.device, self.config['arch_step2']['args']['encoder_type'])
        unseen_class_samples = self.dataset.unseen_class
        unseen_protos_x, unseen_protos_len, unseen_protos_y = get_samples(unseen_class_samples, self.device, self.config['encoder_type'])

        unseen_protos = self.model(unseen_protos_x, unseen_protos_len, 'encode')

        protos, semantic_components = self.model(unseen_protos, seen_class_idxs, 'proto_adapt')

        # protos, memory_protos, novel_protos, after_memory_protos, after_novel_protos, v, loss_r = \
        #     self.model(novel_protos, memory_protos_y, novel_protos_y,'transfer')

        # val seen
        n_val = len(self.dataset.test_seen)
        valid_data_sampler = CLSBatchSampler(range(n_val), self.valid_batch_size)

        for batch_idx, idxs in enumerate(valid_data_sampler):
            query_samples = [self.dataset.test_seen[ids] for ids in idxs]
            querys_x, querys_len, querys_y = get_samples(query_samples, self.device, self.config['encoder_type'])

            # querys = self.model(querys_x, querys_len, memory_protos, novel_protos, after_memory_protos, after_novel_protos, v, 'adapt')

            querys = self.model(querys_x, querys_len, protos, semantic_components, n_seen, 'sample_adapt')

            loss, output = self.loss_fn(protos, querys, querys_y)

            ys = torch.cat([ys, querys_y.cpu()], 0)
            ps = torch.cat([ps, output.cpu()], 0)

        # val unseen
        n_val = len(self.dataset.test_unseen)
        valid_data_sampler = CLSBatchSampler(range(n_val), self.valid_batch_size)

        for batch_idx, idxs in enumerate(valid_data_sampler):
            query_samples = [self.dataset.test_unseen[ids] for ids in idxs]
            querys_x, querys_len, querys_y = get_samples(query_samples, self.device, self.config['encoder_type'])
            querys_y = querys_y + start

            # querys = self.model(querys_x, querys_len, memory_protos, novel_protos,
            #                                        after_memory_protos, after_novel_protos, v, 'adapt')
            querys = self.model(querys_x, querys_len, protos, semantic_components, n_seen, 'sample_adapt')

            loss, output = self.loss_fn(protos, querys, querys_y)

            ys = torch.cat([ys, querys_y.cpu()], 0)
            ps = torch.cat([ps, output.cpu()], 0)

        res, _ = all_metric(ys,
                            ps,
                            len(self.dataset.test_seen),
                            self.dataset.n_seen_class,
                            self.dataset.n_unseen_class)
        # print('\t'.join(['{:.2f}'.format(round(_ * 100, 2)) for _ in res.values()]))

        return res
