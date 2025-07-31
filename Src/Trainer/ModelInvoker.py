"""
created by Allan Yu, 23-11-05
将训练过程、参数、各种组件封装在Trainer类中，便于聚合职责。
"""

import time
from abc import ABC, abstractmethod

import torch.nn as nn

from Utils.Utils import AverageMeter, DynamicMeters
from Utils.Utils import dict2list, message


def default_pack(device, batch):
    return tuple(t.to(device) for t in batch)


class TraningContext(ABC):
    def __init__(self, metric="loss"):
        super().__init__()

        self.use_acc = metric == "acc"
        self.metric = metric

        self.loss = AverageMeter()
        self.acc = DynamicMeters()
        self.last = 0.0
        self.start = time.time()
        self.stop = time.time()
        self.timer_locked = False

    def performance(self):
        if self.use_acc:
            return self.acc.major().avg
        return self.loss.avg

    def elapse(self):
        if self.timer_locked == False:
            self.stop = time.time()
        return self.last + self.stop - self.start

    def lock_timer(self):
        if self.timer_locked == True:
            return
        self.stop = time.time()
        self.last += self.stop - self.start
        self.start = self.stop
        self.timer_locked = True

    def unlock_timer(self):
        self.timer_locked = False
        self.start = time.time()

    def reset(self):
        self.loss.reset()
        self.acc.reset()
        self.last = 0.0
        self.start = time.time()
        self.unlock_timer()

    def avg_loss(self):
        return self.loss.avg

    # TODO: 这个要结合后面的各种info重构下
    def info(self, epoch):
        info = ""
        if self.use_acc:
            if self.acc.num_of_meters() > 1:
                info += f"Acc: {self.acc.major()}[{self.acc}]"
            else:
                info += f"Acc: {self.acc.major()}"

        return info


class ModelInvoker(nn.Module):
    def __init__(self, tm, metric="loss", pack=default_pack, selection=None):
        super().__init__()

        self.tm = tm

        self.metric = metric
        self.pack = pack
        self.selection = selection

        # TODO: 以后有时间，可以做一个可配置的metrics
        # 包含属性(AverageMeter等)、名字（同时对应method）,
        # 计算都是通过batch, results计算，enable acc可以去除了，这个太僵化
        self.enable_acc = (
            # 并非所有的model都需要计算acc
            True
            if hasattr(self, "acc") and callable(getattr(self, "acc", None))
            else False
        )
        # context in training & context in evaluation
        self.trc = TraningContext(metric)
        self.evc = TraningContext(metric)

    ################################
    @property
    def cur_epoch(self):
        return self.tm.current_epoch if self.tm is None else self.tm.cur_epoch

    @property
    def task_name(self):
        return self.tm.task_name

    @property
    def images_dir(self):
        return self.tm.task_images_dir

    ################################
    def training_step(self, batch, batch_idx):
        return self._each_step(self.cur_epoch, self.tm.tr_batch_size, batch_idx, batch, self.trc, False)

    def validation_step(self, batch, batch_idx):
        milestone = self.tm.is_milstone_epoch(self.cur_epoch)
        return self._each_step(self.cur_epoch, self.tm.ev_batch_size, batch_idx, batch, self.evc, milestone)

    def _each_step(self, epoch, batch_size, bidx, batch, context, milestone):
        if self.selection is None:
            batch = self.pack(self.tm.device, batch)
        else:
            batch = dict2list(self.pack(self.tm.device, batch), self.selection)

        results, loss, weight = self.forward(epoch, bidx, batch, milestone)

        context.loss.update(loss.item(), batch_size if weight is None else weight)

        # results是model自己的产物，model自己负责解析
        if self.enable_acc:
            correct, major, weight = self.acc(batch, results)
            context.acc.update(correct, major, batch_size if weight is None else weight)

        return loss

    ################################
    def on_train_epoch_start(self):
        message(self, "begin_of_epoch", epoch=self.cur_epoch)
        self.trc.reset()

    def on_train_epoch_end(self):
        message(self, "end_of_epoch", epoch=self.cur_epoch)
        self.trc.lock_timer()

    def on_validation_start(self):
        message(self, "begin_of_eval", epoch=self.cur_epoch)
        # 避免把eval的时间算入train
        self.trc.lock_timer()
        self.evc.reset()

    def on_validation_end(self):
        message(self, "end_of_eval", epoch=self.cur_epoch)
        self.evc.lock_timer()
        self.trc.unlock_timer()

    def configure_optimizers(self):
        return (
            {
                "optimizer": self.tm.get_optimizer(),
                "lr_scheduler": {"scheduler": self.tm.get_scheduler(), "interval": "step", "frequency": 1},
            },
        )
