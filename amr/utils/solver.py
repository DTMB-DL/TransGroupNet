from amr.utils import logger
import torch
import time
from amr.utils.static import *
from amr.dataloaders.preprocess import *
import os
import numpy as np
__all__ = ["Trainer", "Tester"]


class Trainer:
    def __init__(self, model, device, optimizer, lr_decay, criterion, save_path, valid_freq=1, early_stop=True):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.lr_decay = lr_decay
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=lr_decay,
                                                                    patience=5, verbose=True, threshold=0.0001,
                                                                    threshold_mode='rel', cooldown=0, min_lr=1e-6,
                                                                    eps=1e-08)
        self.criterion = criterion
        self.all_epoch = None
        self.cur_epoch = 1
        self.train_loss = None
        self.train_acc = None
        self.valid_loss = None
        self.valid_acc = None
        self.train_loss_all = []
        self.train_acc_all = []
        self.valid_loss_all = []
        self.valid_acc_all = []
        self.valid_freq = valid_freq
        self.save_path = save_path
        self.best_acc = None
        # early stopping
        self.early_stop = early_stop
        self.patience = 10
        self.delta = 0
        self.counter = 0
        self.stop_flag = False

    def loop(self, epochs, train_loader, valid_loader, eps=1e-5):
        self.all_epoch = epochs
        for ep in range(self.cur_epoch, epochs+1):
            self.cur_epoch = ep
            self.train_loss, self.train_acc = self.train(train_loader)
            self.train_loss_all.append(self.train_loss)
            self.train_acc_all.append(self.train_acc)

            self.valid_loss, self.valid_acc = self.val(valid_loader)
            self.valid_loss_all.append(self.valid_loss)
            self.valid_acc_all.append(self.valid_acc)

            self._loop_postprocessing(self.valid_acc)
            if not self.lr_decay < eps:
                self.scheduler.step(self.valid_loss)

            if self.early_stop and self.stop_flag:
                logger.info(f'early stopping at Epoch: [{self.cur_epoch}]')
                break
        return self.train_loss_all, self.train_acc_all, self.valid_loss_all, self.valid_acc_all

    def train(self, train_loader):
        self.model.train()
        with torch.enable_grad():
            return self._iteration(train_loader)

    def val(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            return self._iteration(val_loader)

    def _iteration(self, data_loader):
        iter_loss = AverageMeter('Iter loss')
        iter_acc = AverageMeter('Iter acc')
        stime = time.time()
        for batch_idx, (X, Y, _) in enumerate(data_loader):
            X, Y = X.to(self.device).float(), Y.to(self.device)
            Y_soft = self.model(X)
            loss, Y_pred = self.criterion(Y_soft, Y, X)

            if self.model.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            acc_pred = (Y_pred == Y).sum()
            acc_total = Y.numel()
            iter_acc.update(acc_pred/acc_total, acc_total)
            iter_loss.update(loss)

        ftime = time.time()
        if self.model.training:
            logger.info(f'Train | '
                        f'Epoch: [{self.cur_epoch}/{self.all_epoch}] | '
                        f'loss: {iter_loss.avg:.3e} | '
                        f'Acc: {iter_acc.avg:.3f} | '
                        f'time: {ftime-stime:.3f}')
        else:
            logger.info(f'Valid | '
                        f'Epoch: [{self.cur_epoch}/{self.all_epoch}] | '
                        f'loss: {iter_loss.avg:.3e} | '
                        f'Acc: {iter_acc.avg:.3f} | '
                        f'time: {ftime-stime:.3f}')

        return iter_loss.avg.item(), iter_acc.avg.item()

    def _save(self, state, name):
        if self.save_path is None:
            logger.warning('No path to save checkpoints.')
            return

        os.makedirs(self.save_path, exist_ok=True)
        torch.save(state, os.path.join(self.save_path, name))

    def _loop_postprocessing(self, acc):
        state = {
            'epoch': self.cur_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_acc': self.best_acc
        }
        if self.best_acc is None:
            self.best_acc = acc
            state['best_acc'] = self.best_acc
            self._save(state, name=f"best_acc.pth")
        elif acc < self.best_acc + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_flag = True
        else:
            self.best_acc = acc
            state['best_acc'] = self.best_acc
            self._save(state, name=f"best_acc.pth")
            self.counter = 0


class Tester:
    def __init__(self, model, device, criterion, classes, snrs):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.classes = classes
        self.snrs = snrs
        self.conf = torch.zeros(classes, classes)
        self.conf_snr = torch.zeros(len(snrs), classes, classes)
        self.acc_pred_snr = torch.zeros(len(snrs))
        self.acc_total_snr = torch.zeros(len(snrs))
        self.acc_snr = torch.zeros(len(snrs))

    def __call__(self, test_loader, verbose=True):
        self.model.eval()
        with torch.no_grad():
            loss, acc = self._iteration(test_loader)

        # 混淆矩阵

        if verbose:
            logger.info(f'Test | '
                        f'loss: {loss:.3e} | '
                        f'Acc: {acc:.3f}')
        return loss, acc, self.conf, self.conf_snr, self.acc_snr

    def _iteration(self, data_loader):
        iter_loss = AverageMeter('Iter loss')
        iter_acc = AverageMeter('Iter acc')
        stime = time.time()
        for batch_idx, (X, Y, Z) in enumerate(data_loader):
            X, Y, Z = X.to(self.device).float(), Y.to(self.device), Z.to(self.device)
            Y_soft = self.model(X)
            loss, Y_pred = self.criterion(Y_soft, Y, X)
            acc_pred = (Y_pred == Y).sum()
            acc_total = Y.numel()
            for i in range(Y.shape[0]):
                self.conf[Y[i], Y_pred[i]] += 1
                idx = self.snrs.index(Z[i])
                self.conf_snr[idx, Y[i], Y_pred[i]] += 1
                self.acc_pred_snr[idx] += (Y[i] == Y_pred[i]).cpu()
                self.acc_total_snr[idx] += 1

            iter_acc.update(acc_pred / acc_total, acc_total)
            iter_loss.update(loss)
        for i in range(self.classes):
            self.conf[i, :] /= torch.sum(self.conf[i, :])
        for j in range(len(self.snrs)):
            self.acc_snr[j] = self.acc_pred_snr[j] / self.acc_total_snr[j]
            for i in range(self.classes):
                self.conf_snr[j, i, :] /= torch.sum(self.conf_snr[j, i, :])

        ftime = time.time()
        logger.info(f'Test | '
                    f'loss: {iter_loss.avg:.3e} | '
                    f'Acc: {iter_acc.avg:.3f} | '
                    f'time: {ftime-stime:.3f}')
        return iter_loss.avg.item(), iter_acc.avg.item()

