
import os
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torchvision.datasets import FashionMNIST
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.functional import accuracy, precision, recall, f1_score


def get_args():
    parser = argparse.ArgumentParser()
    # Init and setup
    parser.add_argument('--seed', type=int,
        help="int. deterministic seed. cudnn.deterministic is always set True by deafult.")
    parser.add_argument('--name', default='default', type=str,
        help="str. default=default. Tensorboard name and log folder name.")
    parser.add_argument('--workers', default=0, type=int,
        help="int. default=0. Dataloader num_workers. good practice is to use number of cpu cores.")
    parser.add_argument('--gpus', nargs="+", default=1, type=int,
        help="str. default=None (cpu). gpus to train on. see pl multi_gpu docs for details.")
    parser.add_argument('--benchmark', default=False, action='store_true',
        help="store_true. set cudnn.benchmark.")
    parser.add_argument('--precision', default=32, type=int, choices=[16, 32],
        help="int. default=32. 32 for full precision and 16 uses pytorch amp")
    # Encoder Parameters
    parser.add_argument('--encoder_dim', default=None, type=int,
        help="int. default=None.")
    # General Training Hyperparameters
    parser.add_argument('--batch', default=8, type=int,
        help="int. default=8. batch size.")
    parser.add_argument('--accumulate', default=1, type=int,
        help="int. default=1. number of gradient accumulation steps. simulate larger batches when >1.")
    parser.add_argument('--epochs', default=1, type=int,
        help="int. deafult=1. number of epochs")
    parser.add_argument('--max_steps', default=-1, type=int,
        help="int. default=-1. optimizer step limit.")
    # Optimizer
    parser.add_argument('--lr', default=1e-5, type=float,
        help="float. default=1e-5. learning rate")
    parser.add_argument('--weight_decay', default=0.0, type=float,
        help="float. default=0.0. weight decay for adamw. 0=no weight decay.")
    parser.add_argument('--adam_b1', default=0.9, type=float)
    parser.add_argument('--adam_b2', default=0.999, type=float)
    # Scheduler
    # Validation
    parser.add_argument('--val_check_interval', default=None, type=int,
        help="int. default=None. number of batchs to check")
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int,
        help="int. default=1.")
    parser.add_argument('--val_percent', default=1.0, type=float,
        help="float. default=1.0. percentage of validation set to test during a validation step.")
    # Misc
    parser.add_argument('--dropout', default=0.0, type=float,
        help="float. default=0.0. Dropout is used before intializing the lstm and before projecting to the vocab.")
    args = parser.parse_args()
    return args

class ClassificationModel(L.LightningModule):
    def __init__(self, encoder, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x):
        """ Inference Method Only"""
        return torch.softmax(self.encoder(x), dim=1)

    def batch_step(self, batch):
        """ Used in train and validation """
        data, target = batch
        data = data.requires_grad_()
        logits = self.encoder(data)
        onehot = F.one_hot(target, num_classes=args.num_classes)
        # target_gradients = 2*onehot - 1
        target_gradients = torch.ones_like(logits)

        # print(f"{data.requires_grad = }")
        # print(f"{target.requires_grad=  }")
        # print(f"{logits.requires_grad = }")
        # print(f"{target_gradients.requires_grad = }")

        # loss = nn.CrossEntropyLoss()(logits, target)
        loss = ((1-onehot)*logits).mean() - ((onehot)*logits).mean()

        if self.training:
            gradient = autograd.grad(
                outputs=logits,
                inputs=data,
                grad_outputs=target_gradients,
                create_graph=True,
                retain_graph=True
            )[0]
            gradient = gradient.view(gradient.size(0), -1)
            """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
            grad_penalty = ((gradient.norm(2, dim=1) - 1) ** 2).mean() 
            loss += 10*grad_penalty
            # logits.backward(target_gradients, retain_graph=True)

        cross_entropy = self.cross_entropy(logits, target)
        pred = torch.argmax(logits, dim=1)
        acc = accuracy(pred, target, task="multiclass", num_classes=self.hparams.num_classes)
        avg_precision = precision(pred, target, task="multiclass", num_classes=self.hparams.num_classes)
        avg_recall = recall(pred, target, task="multiclass", num_classes=self.hparams.num_classes)
        weighted_f1 = f1_score(pred, target, task="multiclass", num_classes=self.hparams.num_classes)
        metrics = {
            "loss": loss,  # attached to computation graph
            "accuracy": acc,
            "error": 1-acc,
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "weighted_f1": weighted_f1,
            "inv_f1": 1-weighted_f1,
            "cross_entropy": cross_entropy,
        }
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self.batch_step(batch)
        for k, v in metrics.items():
            key = "{}/train".format(k)
            self.log(key, v, on_step=True, on_epoch=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.logger.experiment.add_scalar('Learning Rate/step', lr, global_step=self.global_step)
        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        metrics = self.batch_step(batch)
        for k, v in metrics.items():
            key = "{}/val_epoch".format(k)
            self.log(key, v, on_step=False, on_epoch=True)
        return metrics["loss"]

    def configure_optimizers(self):
        params = self.encoder.parameters()
        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.lr,
            betas=(self.hparams.adam_b1, self.hparams.adam_b2),
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

def main(args):

    L.seed_everything(args.seed)

    # Increment to find the next availble name
    logger = TensorBoardLogger(save_dir="logs", name=args.name)    
    dirpath = f"logs/fashion-{args.name}/version_{logger.version}"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
    args.classes = [
        "T-shirt/Top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat", 
        "Sandal", 
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle Boot",
    ]
    args.num_classes = len(args.classes)

    encoder = nn.Sequential(
        nn.BatchNorm2d(1),
        nn.Conv2d(1, 32, kernel_size=3, bias=False, padding=1),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.GELU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 64, kernel_size=3, bias=False),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.GELU(),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 128, kernel_size=3, bias=False),
        nn.GELU(),
        nn.Flatten(),
        nn.Dropout(args.dropout),
        nn.Linear(128*4*4, 128),
        nn.GELU(),
        nn.BatchNorm1d(128),
        nn.Linear(128, 128),
        nn.GELU(),
        nn.Linear(128, args.num_classes),
    )

    train_ds = FashionMNIST("./data/", train=True, transform=T.ToTensor(), download=True)
    val_ds = FashionMNIST("./data/", train=False, transform=T.ToTensor(), download=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
        num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
        num_workers=0, pin_memory=True)

    model = ClassificationModel(
        encoder=encoder,
        **vars(args)
    )

    trainer = L.Trainer(
        accumulate_grad_batches=args.accumulate,
        benchmark=args.benchmark,  # cudnn.benchmark
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        deterministic=True,  # cudnn.deterministic
        accelerator="gpu",
        devices=args.gpus,
        logger=logger,
        precision=args.precision,
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        num_sanity_val_steps=0,
        limit_val_batches=args.val_percent,
        log_every_n_steps=1,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

if __name__ == "__main__":
    args = get_args()
    main(args)
