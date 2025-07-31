"""
created by Allan Yu, 23-11-05
将训练过程、参数、各种组件封装在Trainer类中，便于聚合职责。
"""

#!/usr/bin/env python3

import math
import os
import random

from datetime import datetime

import numpy
import torch
import torch.nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

########################################################################
from Utils.Utils import extract, readable_time
from Utils.Utils import weight_init, load_model, save_model

from LocalOnly.PathConfig import path_config


class Trainer:
    def __init__(self, task, taskinfo, **extargs):
        self.start_time = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        print(self.start_time)

        ################################################################
        # 目录参数应依赖于系统，存于path_config
        #
        # 除此之外绝大部分参数优先级为path_config < task < extargs
        # 这也可以视为从普遍到特殊的顺序，
        # path_config适用于全局，task适用于任务，extrags仅适用于本次运行
        args = self.args = {}

        ################################################################
        # 目录
        """根据PathConfigTemplate.py制作当前系统的LocalOnly/PathConfig.py，
        该文件不会上传到git，因此是系统独立适用的。"""
        self.localonly_args_text = ""
        for key, value in path_config.items():
            self.localonly_args_text += key + " " + str(value) + "\n"
        args.update(path_config)

        ## 输出目录
        self.path_logs = extract(args, "path_logs", "../logs")
        self.path_record = extract(args, "path_record", "../result-records")
        self.path_chkpt = extract(args, "path_chkpt", "../checkpoints")
        self.path_images = extract(args, "path_images", "../images")

        ## 输入目录
        self.path_initmodel = extract(args, "path_initmodel", "../initmodels")  # 重要的模型参数
        self.task_dir = extract(args, "task_dir", "Tasks")

        # 任务参数
        self.task = task
        self.taskinfo = taskinfo

        # 任务统一的参数（如batch_size）、必要的依赖task的参数(如betterify)，避免task重复设置
        task_dep_args = self.taskinfo["args"]
        self.task_dep_args_text = ""
        for key, value in task_dep_args.items():
            self.task_dep_args_text += key + " " + str(value) + "\n"
        args.update(task_dep_args)

        ################################################################
        # 任务运行参数，为最高优先级
        self.args_text = ""
        for key, value in extargs.items():
            self.args_text += key + " " + str(value) + "\n"
        args.update(extargs)

        ################################################################
        # 随机性设置
        self.seed = extract(args, "seed", 42)
        self.initialize_randomness()

        ################################################################
        # 任务
        self.task_id = f"[{self.start_time}] {self.task}"
        self.task_abstract = extract(args, "abstract", "")
        self.task_description = extract(args, "description", "")
        self.task_name = f"{self.task_id} {self.task_abstract}"
        self.task_name += f" {self.task_description}" if self.task_description != "" else ""
        self.prepare_task_workspace()

        ################################################################
        # 训练策略
        ## 设备
        self.device = torch.device(extract(args, "device", "cuda" if torch.cuda.is_available() else "cpu"))

        ## workers，这个和CPU有关
        self.num_workers = extract(args, "num_workers", 8)
        self.num_workers = self.num_workers

        ################################################################
        # 数据集
        self.tr_batch_size = extract(args, "tr_batch_size", 128)
        self.ev_batch_size = extract(args, "ev_batch_size", self.tr_batch_size)
        self.prepare_dataset_loader()

        self.tr_samples = len(self.tr_loader.dataset)
        self.ev_samples = len(self.ev_loader.dataset)
        self.tr_batches = len(self.tr_loader)
        self.ev_batches = len(self.ev_loader)

        ################################################################
        # 网络
        ## 加载chkpt
        # skip_epochs是从0计数，因此，要从哪个epoch开始就填哪个epoch，一般是存档的下一个epoch
        self.apply_weight_init = extract(args, "apply_weight_init", False)
        self.reload = extract(args, "reload", False)
        self.chkpt = extract(args, "chkpt", None)
        self.ignore_keys = extract(args, "ignore_keys", None)
        self.cur_epoch = -1

        self.prepare_model()

        ################################################################
        # 迭代配置
        self.epochs = extract(args, "epochs", 65)

        self.batchs_in_step = extract(args, "batchs_in_step", 1)
        self.samples_in_step = self.tr_batch_size * self.batchs_in_step

        self.eval_per_train = extract(args, "eval_per_train", 1.0)
        # 浮点数代表比例，且最大为1.0(不能跨epoch)；
        # 若不是浮点数是，代表数量。更建议用比例
        if isinstance(self.eval_per_train, float) and self.eval_per_train <= 1.0:
            self.samples_in_phase = self.tr_samples * self.eval_per_train
        else:
            self.samples_in_phase = self.eval_per_train

        # batch数量取ceil，不足一个batch按一个算
        self.batchs_in_phase = math.ceil(float(self.samples_in_phase) / self.tr_batch_size)
        self.batchs_in_phase = min(self.tr_batches, self.batchs_in_phase)

        # 基础单位是sample，所有数量以此为基础
        # sample->iter(batch)->step->phase->epoch
        self.steps_in_epoch = self.tr_batches // self.batchs_in_step
        self.phases_in_epoch = self.tr_batches // self.batchs_in_phase

        self.tr_batchs_gpu = self.tr_batches
        self.ev_batchs_gpu = self.ev_batches

        ################################################################
        # 优化
        self.lr = extract(args, "lr", 1e-1)
        self.lr_weight_decay = extract(args, "lr_weight_decay", 5e-4)
        self.lr_max_grad_norm = extract(args, "lr_max_grad_norm", 1.0)

        self.prepare_optimizer()

        ################################################################
        # 学习调度
        samples_per_opt = self.samples_in_step
        self.sch_epoch = extract(args, "sch_epoch", [10, 15, 20, 25])
        self.sch_steps = [sch * self.tr_samples // samples_per_opt for sch in self.sch_epoch]
        print(self.sch_steps)
        self.sch_steps = extract(args, "sch_steps", self.sch_steps)
        self.sch_gamma = extract(args, "sch_gamma", 0.2)

        self.prepare_scheduler()

        ################################################################
        # 记录
        self.chkpt_milestones = extract(args, "chkpt_milestones", [61])
        self.comments = extract(args, "comments", "")  # 这是大段的记录，比description详细
        self.best_performance = extract(args, "best_performance", 0.6)
        self.betterify = extract(args, "betterify", "min")
        self.prepare_recoder()

    def initialize_randomness(self):
        random.seed(self.seed)
        numpy.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        #
        cudnn.benchmark = False
        cudnn.deterministic = True

    def prepare_task_workspace(self):
        # self.path_chkpt = os.path.join(self.path_chkpt, str(0))
        if not os.path.exists(self.path_chkpt):
            os.makedirs(self.path_chkpt)

        # self.path_record = os.path.join(self.path_record, str(0))
        if not os.path.exists(self.path_record):
            os.mkdir(self.path_record)

        # use tensorboard
        if not os.path.exists(self.path_logs):
            os.mkdir(self.path_logs)

        # image
        if not os.path.exists(self.path_images):
            os.mkdir(self.path_images)

        if not os.path.exists(self.task_images_dir):
            os.mkdir(self.task_images_dir)

        images_all_dir = os.path.join(self.task_images_dir, "all")
        if not os.path.exists(images_all_dir):
            os.mkdir(images_all_dir)

    def prepare_dataset_loader(self):
        dataset_v = self.taskinfo["create_dataloader"](self)
        dataset_t = self.taskinfo["create_dataloader"](self, is_train=True)

        self.tr_loader = DataLoader(
            dataset_t,
            batch_size=self.tr_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        self.ev_loader = DataLoader(
            dataset_v,
            batch_size=self.ev_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def prepare_model(self):
        self.model = self.taskinfo["create_model"](self)

        if self.apply_weight_init:
            self.model.apply(weight_init)

        if self.reload:
            search_dirs = ["", self.path_chkpt, self.path_initmodel]
            for dir in search_dirs:
                path = os.path.join(dir, self.chkpt)
                if os.path.exists(path):
                    break
            else:
                raise RuntimeError(f"invalid chkpt path: {self.chkpt}\n")

            load_model(self.model, path=path, ignore_keys=self.ignore_keys)

        self.model.to(self.device)

    def prepare_optimizer(self):
        print(
            "Total num. of parameters: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        )
        # optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.lr_weight_decay
        )

    def prepare_scheduler(self):
        # scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.sch_steps, gamma=self.sch_gamma
        )

    @torch.no_grad()
    def eval_phase(self, epoch, milestone=False):
        self.on_validation_start(epoch)
        self.model.on_validation_start()

        self.model.eval()
        for bidx, batch in enumerate(self.ev_loader):
            self.model.validation_step(batch, bidx)
            self.on_validation_batch_end(epoch, bidx)

        self.model.on_validation_end()

    def train_phase(self, epoch):
        self.on_train_epoch_start(epoch)

        self.model.on_train_epoch_start()
        self.model.train()

        self.optimizer.zero_grad()
        for bidx, batch in enumerate(self.tr_loader):
            loss = self.model.training_step(batch, bidx)

            loss = loss / self.batchs_in_step
            loss.backward()

            if (bidx + 1) % self.batchs_in_step == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.lr_max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            self.on_train_batch_end(epoch, bidx)

            if (bidx + 1) % self.batchs_in_phase == 0:
                yield loss
                self.model.train()

        self.model.on_train_epoch_end()
        self.on_train_epoch_end(epoch)

    def prepare_recoder(self):
        recorder = os.path.join(self.path_record, self.task_name + ".txt")
        file = open(recorder, "a")

        file.write("[abstract]\n" + self.task_abstract + "\n")
        if self.task_description != "":
            file.write("[description]\n" + self.task_description + "\n")
        if self.comments != "":
            file.write("[comments]\n" + self.comments + "\n")
        file.write("[args]\n" + self.args_text + "\n")
        file.write("[task dep args]\n" + self.task_dep_args_text + "\n")
        file.write("[localonly args]\n" + self.localonly_args_text + "\n")
        file.write("[model]\n" + str(self.model) + "\n")

        file.write("epoch \t[T][E]loss \t[T][E]consumed time \t[T][E]extra information\n")
        file.close()

        import shutil

        shutil.copy(recorder, os.path.join(self.task_images_dir, "config.txt"))

    def is_milstone_epoch(self, epoch):
        return epoch in self.chkpt_milestones

    @property
    def task_images_dir(self):
        return os.path.join(self.path_images, "images-" + self.task_name)

    def get_optimizer(self):
        return self.optimizer

    def get_scheduler(self):
        return self.scheduler

    def record_result(self, epoch, t_loss, e_loss, t_time, e_time, t_info, e_info, bsf=False):
        file = self.task_name + ".txt"
        file = open(os.path.join(self.path_record, file), "a")

        file.write(str(epoch) + "\t")

        file.write("[T]{:0.4f}".format(t_loss))
        file.write("[E]{:0.4f}".format(e_loss) + "\t")

        file.write("[T]{}".format(readable_time(t_time)))
        file.write("[E]{}".format(readable_time(e_time)) + "\t")

        file.write("[T]" + t_info if t_info != "" else "")
        file.write("[E]" + e_info + "\t" if e_info != "" else "")

        file.write("BSF\n" if bsf else "\n")
        file.close()

    def save_weight(self, epoch=None, best=None, filename=None, ignore_keys=None):
        if filename is None:
            filename = self.task_name
            if epoch is not None:
                filename += f"-e{epoch}"
            if best is not None:
                filename += "-{:.4f}".format(best)
            filename += f".pth"

        weights_path = os.path.join(self.path_chkpt, filename)
        print("saving weights file to " + weights_path)

        save_model(self.model, path=weights_path, ignore_keys=ignore_keys)

    def checkpt(self, epoch, performance):
        if self.betterify == "min":
            bsf = self.best_performance > performance
        else:
            bsf = self.best_performance < performance

        if bsf:
            self.best_performance = performance
            self.save_weight(epoch=epoch, best=self.best_performance)
        elif epoch in self.chkpt_milestones:
            self.save_weight(epoch=epoch)

        return bsf

    def train(self):
        for epoch in range(self.epochs):
            for phase, _ in enumerate(self.train_phase(epoch)):
                milestone = epoch in self.chkpt_milestones

                self.eval_phase(epoch, milestone=milestone)

    def test(self):
        # test没有epoch概念，epoch统一设置成-1
        self.chkpt_milestones = [-1]
        self.eval_phase(-1, milestone=True)
        loss, time, info = self.model.evc.avg_loss(), self.model.evc.elapse(), self.model.evc.info(-1)

        print("\r" + " " * 120, end="\r")

        print(f"Time Consumed: {readable_time(time)}    ", end="")
        print(f"Average loss: {loss:.4f}    ", end="")
        print(f"{info}" if info != "" else "")

    # 训练事件，为适配lightning而建立的事件
    def on_train_epoch_start(self, epoch):
        self.cur_epoch = epoch

        print("\r" + " " * 120, end="")
        print(f"\r[{epoch+1}/{self.epochs}]Tr. epoch {epoch}>>> ", end="")

    def on_validation_start(self, epoch):
        print("\r" + " " * 120, end="")
        print(f"\r[{epoch+1}/{self.epochs}]Ev. epoch {epoch}>>> ", end="")

    # 为兼容lightning，提取的事件处理
    def on_train_batch_end(self, epoch, bidx):
        loss, time, info = self.model.trc.avg_loss(), self.model.trc.elapse(), self.model.trc.info(epoch)
        text = f"\r[{epoch+1}/{self.epochs}]Tr. epoch {epoch}>>> "
        text += "{:2.0%}[{}/{}]    {} Consumed    LR: {:0.6f}    Loss: {:0.4f}{} ".format(
            bidx / self.tr_batchs_gpu,
            bidx,
            self.tr_batchs_gpu,
            readable_time(time),
            self.optimizer.param_groups[0]["lr"],
            loss,
            "    " + info if info != "" else "",
        )
        print(text, end="")

    def on_validation_batch_end(self, epoch, bidx):
        loss, time, info = self.model.evc.avg_loss(), self.model.evc.elapse(), self.model.evc.info(epoch)
        text = f"\r[{epoch+1}/{self.epochs}]Ev. epoch {epoch}>>> "
        text += "{:2.0%}[{}/{}]    {} Consumed    Loss: {:0.4f}{} ".format(
            bidx / self.ev_batchs_gpu,
            bidx,
            self.ev_batchs_gpu,
            readable_time(time),
            loss,
            "    " + info if info != "" else "",
        )
        print(text, end="")

    def on_train_epoch_end(self, epoch):
        performance = self.model.evc.performance()
        bsf = self.checkpt(epoch, performance)

        # 为了公平起见，这里t_time是包含e_time的，所以重新计算
        t_loss, e_loss = self.model.trc.avg_loss(), self.model.evc.avg_loss()
        t_time, e_time = self.model.trc.elapse(), self.model.evc.elapse()
        t_info, e_info = self.model.trc.info(epoch), self.model.evc.info(epoch)

        text = f"epoch {epoch}>>> "
        text += f"[T]{readable_time(t_time)}/[E]{readable_time(e_time)} Consumed    "
        text += "LR: {:0.6f}    ".format(self.optimizer.param_groups[0]["lr"])
        text += "Loss: [T]{:.4f}/[E]{:.4f}    ".format(t_loss, e_loss)
        text += f"[T]{t_info}/" if t_info != "" else ""
        text += f"[E]{e_info} " if e_info != "" else ""
        text += " BSF" if bsf else ""

        print("\r" + " " * 120, end="\r")
        print(text)

        self.record_result(epoch, t_loss, e_loss, t_time, e_time, t_info, e_info, bsf=bsf)
