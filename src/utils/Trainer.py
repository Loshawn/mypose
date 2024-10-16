import os
from src.utils.logger import Logger
from src import models, data
import torch
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
from torch.amp import GradScaler, autocast
from tqdm import tqdm


class Trainer():

    def __init__(self, args, datasets, model, optimizer, criterion):
        self.args = args
        self.distributed = args.distributed

        self.loader_train, self.loader_val = data.build_dataloader(
            args, datasets[0], datasets[1])
        self.optimizer, self.criterion = optimizer, criterion

        milestones = args.lr_config['step']
        gamma = 0.1
        self.scheduler = MultiStepLR(optimizer, milestones, gamma)

        self.global_step = 0
        self.num_warmup_steps = args.lr_config[
            'warmup_iters']  # Number of warm-up steps
        warmup_factor = args.lr_config[
            'warmup_ratio']  # Initial learning rate = warmup_factor * learning_rate
        self.warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: warmup_factor +
            (1.0 - warmup_factor) * step / self.num_warmup_steps)

        if self.distributed:
            self.rank = dist.get_rank()
            self.sampler = self.loader_train.sampler
            self.device = f'cuda:{self.rank}'
            model = model.to(self.device)
            self.model = DDP(model, device_ids=[self.rank])

        else:
            self.rank = 0
            self.device = f'cuda:{self.rank}'
            self.model = model.to(self.device)

        self.logger = Logger(args, rank=self.rank)

        if args.resume != '.':
            path = f"{self.args.resume}/model"
            self.model = models.load_model(self.model, f"{path}/model.pth",
                                           self.device)

            self.optimizer, self.logger.info = models.load_optimizer_and_logger(
                self.optimizer, f"{path}/optimizer.pth", self.device)
        self.start_epoch = self.logger.info["Epoch"] + 1

        if args.use_amp:
            self.scaler = GradScaler()

    def train(self):
        self.logger.add_log(f"Train :")
        losses = 0
        total_samples = 0

        self.model.train()

        if self.distributed:
            self.sampler.set_epoch(self.logger.info["Epoch"])

        loop = tqdm(self.loader_train, total=len(self.loader_train),
                    ncols=100) if self.rank == 0 else self.loader_train

        for images, targets, target_weights, _ in loop:
            self.optimizer.zero_grad()

            images, targets, target_weights = images.to(
                self.device), targets.to(self.device), target_weights.to(
                    self.device)

            if self.args.use_amp:
                with autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets, target_weights)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)  #
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets, target_weights)
                loss.backward()
                self.optimizer.step()

            if self.global_step < self.num_warmup_steps:
                self.warmup_scheduler.step()
            self.global_step += 1

            losses += loss.item()
            total_samples += images.size(0)

            if self.rank == 0:
                loop.set_postfix(loss=losses / total_samples)

        self.scheduler.step()
        if self.distributed:
            losses_tensor = torch.tensor(losses).to(self.device)
            total_samples_tensor = torch.tensor(total_samples).to(self.device)
            dist.all_reduce(losses_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

            avg_loss = losses_tensor.item() / total_samples_tensor.item()
        else:
            avg_loss = losses / total_samples

        self.logger.info['train_loss'].append(avg_loss)
        self.logger.add_log(f"\tLoss : {avg_loss:.5e}\n")

        self.logger.printf(f"Train Loss : {avg_loss}")

    def test(self):
        self.logger.add_log(f"Test :")
        losses = 0
        acc_samples = 0
        total_samples = 0
        self.model.eval()

        loop = tqdm(self.loader_val, total=len(self.loader_val),
                    ncols=100) if self.rank == 0 else self.loader_val

        with torch.no_grad():
            for images, targets, target_weights, _ in loop:
                images, targets, target_weights = images.to(
                    self.device), targets.to(self.device), target_weights.to(
                        self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets, target_weights)
                if self.distributed:
                    acc = self.model.module.keypoint_head.get_accuracy(
                        outputs, targets, target_weights)['acc_pose']
                else:
                    acc = self.model.keypoint_head.get_accuracy(
                        outputs, targets, target_weights)['acc_pose']

                losses += loss.item()
                acc_samples += images.size(0) * acc
                total_samples += images.size(0)

                if self.rank == 0:
                    loop.set_postfix({
                        "loss": losses / total_samples,
                        "acc": acc_samples / total_samples
                    })

        if self.distributed:
            losses_tensor = torch.tensor(losses).to(self.device)
            acc_samples_tensor = torch.tensor(acc_samples).to(self.device)
            total_samples_tensor = torch.tensor(total_samples).to(self.device)
            dist.all_reduce(losses_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(acc_samples_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

            avg_loss = losses_tensor.item() / total_samples_tensor.item()
            acc = acc_samples_tensor.item() / total_samples_tensor.item()
        else:
            avg_loss = losses / total_samples

        self.logger.info['test_loss'].append(avg_loss)
        self.logger.info['acc'].append(acc)
        self.logger.draw_loss()
        self.logger.add_log(f"\tLoss : {avg_loss:.5e}")
        self.logger.add_log(f"\tAcc : {acc:.5f}\n")

        self.logger.printf(f"Test Loss : {avg_loss}")
        self.logger.printf(f"Acc : {acc}")

    def begin(self):
        self.logger.printf("Beging Training ...")

        for epoch in range(self.start_epoch, self.args.epochs + 1):
            self.logger.printf(f'\nEpoch [{epoch}/{self.args.epochs}]')

            lr = self.optimizer.param_groups[0]['lr']
            self.logger.add_log(f"[Epoch: {epoch} ] : Learning rate : {lr}")

            self.train()
            self.test()

            if self.rank == 0:
                models.save_model(
                    self.model, f"{self.logger.dir_model}/model_{epoch}.pth")
                models.save_optimizer_and_logger(
                    self.optimizer, self.logger.info,
                    f"{self.logger.dir_model}/optimizer.pth")
                
            self.logger.info["Epoch"] = epoch

        self.logger.printf(
            f"------------------------------\n"
            f"Training complete\n"
            f"The trained model weights are saved at: \n{self.logger.dir_model}"
        )
