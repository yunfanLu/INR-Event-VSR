import os
import random
import shutil
import time
from collections import OrderedDict
from os.path import join, isfile

import cv2
import numpy as np
import torch
import torch.nn as nn
from absl.logging import flags
from absl.logging import info
from torch.testing._internal.common_quantization import AverageMeter
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from egvsr.core.optimizer import Optimizer
from egvsr.datasets import get_dataset
from egvsr.losses import get_metric, get_loss
from egvsr.models import get_model

FLAGS = flags.FLAGS


class Visualization:
    def __init__(self, visualization_config):
        self.saving_folder = join(FLAGS.log_dir, visualization_config.folder)
        os.makedirs(self.saving_folder, exist_ok=True)
        self.count = 0

        self.tag = visualization_config.tag
        info("Init Visualization:")
        info(f"  saving_folder: {self.saving_folder}")

    def visualize(self, lrs, srs, hrs, lr_events, hr_event):
        n, _, _, _ = hrs.shape
        for i in range(n):
            lr, sr, hr, event = lrs[i], srs[0][i], hrs[i], lr_events[i]
            lr = lr.permute(1, 2, 0).cpu().numpy() * 255
            sr = sr.permute(1, 2, 0).cpu().numpy() * 255
            hr = hr.permute(1, 2, 0).cpu().numpy() * 255
            event = event.permute(1, 2, 0).cpu().numpy()
            event = np.sum(event, axis=2)
            event_image = np.zeros((event.shape[0], event.shape[1], 3)) + 255
            event_image[event > 0] = [0, 0, 255]
            event_image[event < 0] = [255, 0, 0]
            event_image = event_image.astype(np.uint8)

            lr = cv2.cvtColor(lr, cv2.COLOR_GRAY2BGR)
            sr = cv2.cvtColor(sr, cv2.COLOR_GRAY2BGR)
            hr = cv2.cvtColor(hr, cv2.COLOR_GRAY2BGR)
            index = str(self.count).zfill(5)
            lr_path = join(self.saving_folder, f"{index}_{self.tag}_lr.png")
            sr_path = join(self.saving_folder, f"{index}_{self.tag}_sr.png")
            hr_path = join(self.saving_folder, f"{index}_{self.tag}_hr.png")
            event_path = join(self.saving_folder, f"{index}_{self.tag}_event.png")
            cv2.imwrite(filename=lr_path, img=lr)
            cv2.imwrite(filename=sr_path, img=sr)
            cv2.imwrite(filename=hr_path, img=hr)
            cv2.imwrite(filename=event_path, img=event_image)
            self.count += 1


class ParallelLaunch:
    def __init__(self, config):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6666"
        info(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        info(f"MASTER_PORT: {os.environ['MASTER_PORT']}")
        # 0. config
        self.config = config
        # # 1. init environment
        # torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True
        # 1.1 init global random seed
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)
        random.seed(config.SEED)
        np.random.seed(config.SEED)
        # 1.2 init the tensorboard log dir
        self.tb_recoder = SummaryWriter(FLAGS.log_dir)
        # 2. device
        self.visualizer = None
        if config.VISUALIZE:
            self.visualizer = Visualization(config.VISUALIZATION)

    def run(self):
        # 0. Init
        train_dataset, val_dataset = get_dataset(self.config.DATASET)
        model = get_model(self.config.MODEL)
        criterion = get_loss(self.config.LOSS)
        metrics = get_metric(self.config.METRICS)
        opt = Optimizer(self.config.OPTIMIZER, model)
        # 1. Build model
        if self.config.IS_CUDA:
            model = nn.DataParallel(model)
            model = model.cuda()

        if self.config.RESUME.PATH:
            if not isfile(self.config.RESUME.PATH):
                raise ValueError(f"File not found, {self.config.RESUME.PATH}")
            if self.config.IS_CUDA:
                checkpoint = torch.load(
                    self.config.RESUME.PATH,
                    map_location=lambda storage, loc: storage.cuda(0),
                )
            else:
                checkpoint = torch.load(self.config.RESUME.PATH, map_location=torch.device("cpu"))
                new_state_dict = OrderedDict()
                for k, v in checkpoint["state_dict"].items():
                    name = k[7:]
                    new_state_dict[name] = v
                checkpoint["state_dict"] = new_state_dict

            if self.config.RESUME.SET_EPOCH:
                self.config.START_EPOCH = checkpoint["epoch"]
                opt.optimizer.load_state_dict(checkpoint["optimizer"])
                opt.scheduler.load_state_dict(checkpoint["scheduler"])

            model.load_state_dict(checkpoint["state_dict"])
        # 2. Build Dataloader
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.JOBS,
            pin_memory=False,
            drop_last=True,
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.JOBS,
            pin_memory=False,
            drop_last=True,
        )
        # 3. if test only
        if self.config.TEST_ONLY:
            self.valid(val_loader, model, criterion, metrics, 0)
            return
        # 4. train
        min_loss = 123456789.0
        for epoch in range(self.config.START_EPOCH, self.config.END_EPOCH):
            self.train(train_loader, model, criterion, metrics, opt, epoch)
            # save checkpoint
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": opt.optimizer.state_dict(),
                "scheduler": opt.scheduler.state_dict(),
            }
            path = join(self.config.SAVE_DIR, "checkpoint.pth.tar")
            time.sleep(1)
            # valid
            if epoch % self.config.VAL_INTERVAL == 0:
                torch.save(checkpoint, path)
                val_loss = self.valid(val_loader, model, criterion, metrics, epoch)
                if val_loss < min_loss:
                    min_loss = val_loss
                    copy_path = join(self.config.SAVE_DIR, "model_best.pth.tar")
                    shutil.copy(path, copy_path)
            # train
            if epoch % self.config.MODEL_SANING_INTERVAL == 0:
                path = join(
                    self.config.SAVE_DIR,
                    f"checkpoint-{str(epoch).zfill(3)}.pth.tar",
                )
                torch.save(checkpoint, path)

    def train(self, train_loader, model, criterion, metrics, opt, epoch):
        model.train()
        info(f"Train Epoch[{epoch}]:len({len(train_loader)})")
        length = len(train_loader)
        # 1. init meter
        losses_meter = {"TotalLoss": AverageMeter(f"Valid/TotalLoss")}
        for config in self.config.LOSS:
            losses_meter[config.NAME] = AverageMeter(f"Train/{config.NAME}")
        metric_meter = {}
        for config in self.config.METRICS:
            metric_meter[config.NAME] = AverageMeter(f"Train/{config.NAME}")
        batch_time_meter = AverageMeter("Train/BatchTime")
        # 2. start a training epoch
        start_time = time.time()
        time_recoder = time.time()
        for index, (lr, lr_event, sr, hr_event) in enumerate(train_loader):
            if self.config.IS_CUDA:
                lr = lr.cuda()
                lr_event = lr_event.cuda()
                sr = sr.cuda()
                hr_event = hr_event.cuda()
            outputs = model(lr, lr_event)
            losses, name_to_loss = criterion(outputs, sr, lr, lr_event, hr_event)
            # 2.1 forward
            name_to_measure = metrics(outputs, sr, lr, lr_event, hr_event)
            # 2.2 backward
            opt.zero_grad()
            losses.backward()
            # 2.3 update weights
            # clip the grad
            # clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            opt.step()
            # 2.4 update measure
            # 2.4.1 time update
            now = time.time()
            batch_time_meter.update(now - time_recoder)
            time_recoder = now
            # 2.4.2 loss update
            losses_meter["TotalLoss"].update(losses.detach().item())
            for name, loss_item in name_to_loss:
                loss_item = loss_item.detach().item()
                losses_meter[name].update(loss_item)
            # 2.4.3 measure update
            for name, measure_item in name_to_measure:
                measure_item = measure_item.detach().item()
                metric_meter[name].update(measure_item)
            # 2.5 log
            if index % self.config.LOG_INTERVAL == 0:
                info(f"Train Epoch[{epoch}, {index}/{length}]:")
                for name, meter in losses_meter.items():
                    info(f"    loss:    {name}: {meter.avg}")
                for name, measure in metric_meter.items():
                    info(f"    measure: {name}: {measure.avg}")
        # 3. record a training epoch
        # 3.1 record epoch time
        epoch_time = time.time() - start_time
        batch_time = batch_time_meter.avg
        info(f"Train Epoch[{epoch}]:time:epoch({epoch_time}),batch({batch_time})" f"lr({opt.get_lr()})")
        self.tb_recoder.add_scalar(f"Train/EpochTime", epoch_time, epoch)
        self.tb_recoder.add_scalar(f"Train/BatchTime", batch_time, epoch)
        self.tb_recoder.add_scalar(f"Train/LR", opt.get_lr(), epoch)
        for name, meter in losses_meter.items():
            info(f"    loss:    {name}: {meter.avg}")
            self.tb_recoder.add_scalar(f"Train/{name}", meter.avg, epoch)
        for name, measure in metric_meter.items():
            info(f"    measure: {name}: {measure.avg}")
            self.tb_recoder.add_scalar(f"Train/{name}", measure.avg, epoch)
        # adjust learning rate
        opt.lr_schedule()

    def valid(self, valid_loader, model, criterion, metrics, epoch):
        length = len(valid_loader)
        info(f"Valid Epoch[{epoch}] starting: length({length})")
        model.eval()
        with torch.no_grad():
            # 1. init meter
            losses_meter = {"total": AverageMeter(f"Valid/TotalLoss")}
            for config in self.config.LOSS:
                losses_meter[config.NAME] = AverageMeter(f"Valid/{config.NAME}")
            metric_meter = {}
            for config in self.config.METRICS:
                metric_meter[config.NAME] = AverageMeter(f"Valid/{config.NAME}")
            batch_time_meter = AverageMeter("Valid/BatchTime")
            # 2. start a validating epoch
            time_recoder = time.time()
            start_time = time_recoder
            for index, (lr, lr_event, sr, hr_event) in enumerate(valid_loader):
                if self.config.IS_CUDA:
                    lr = lr.cuda()
                    lr_event = lr_event.cuda()
                    sr = sr.cuda()
                    hr_event = hr_event.cuda()
                outputs = model(lr, lr_event)
                losses, name_to_loss = criterion(outputs, sr, lr, lr_event, hr_event)
                # 2.2. recorder
                name_to_measure = metrics(outputs, sr, lr, lr_event, hr_event)
                # 2.3 visualization
                if self.visualizer:
                    self.visualizer.visualize(lr, outputs, sr, lr_event, hr_event)
                # 2.4. update measure
                now = time.time()
                batch_time_meter.update(now - time_recoder)
                time_recoder = now
                losses_meter["total"].update(losses.detach().item())
                for name, loss_item in name_to_loss:
                    loss_item = loss_item.detach().item()
                    losses_meter[name].update(loss_item)
                for name, measure_item in name_to_measure:
                    measure_item = measure_item.detach().item()
                    metric_meter[name].update(measure_item)
                if index % self.config.LOG_INTERVAL == 0:
                    info(f"Valid Epoch[{epoch}, {index}/{length}]:")
                    info(f"    batch-time: {batch_time_meter.avg}")
                    for name, meter in losses_meter.items():
                        info(f"    loss:    {name}: {meter.avg}")
                    for name, measure in metric_meter.items():
                        info(f"    measure: {name}: {measure.avg}")
            # 3. record a training epoch
            # 3.1 record epoch time
            epoch_time = time.time() - start_time
            batch_time = batch_time_meter.avg
            info(f"Valid Epoch[{epoch}]:" f"time:epoch({epoch_time}),batch({batch_time})")
            self.tb_recoder.add_scalar(f"Valid/EpochTime", epoch_time, epoch)
            self.tb_recoder.add_scalar(f"Valid/BatchTime", batch_time, epoch)
            for name, meter in losses_meter.items():
                info(f"    loss:    {name}: {meter.avg}")
                self.tb_recoder.add_scalar(f"Valid/{name}", meter.avg, epoch)
            for name, measure in metric_meter.items():
                info(f"    measure: {name}: {measure.avg}")
                self.tb_recoder.add_scalar(f"Valid/{name}", measure.avg, epoch)
            return losses_meter["total"].avg
