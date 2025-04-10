# -*- coding: utf-8 -*-
"""

CUIテンプレート

ToDo
- 実装する
- testする

@author: tadahaya
"""
# packages installed in the current environment
import os, datetime, argparse, time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# original packages in src
from .src import utils
from .src import data_handler as dh
from .src.trainer import Trainer
from .src.models import MyNet

# === 基本的にタスクごとに変更 ===
# argumentの設定, 概ね同じセッティングの中で振りうる条件を設定
parser = argparse.ArgumentParser(description='CLI')
parser.add_argument(
    'workdir',
    type=str,
    help='working directory that contains the dataset'
    )
parser.add_argument('--note', type=str, help='short note for this running')
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--num_epochs', type=int, default=5) # epoch
parser.add_argument('--batch_size', type=int, default=128) # batch size
parser.add_argument('--lr', type=float, default=0.001) # learning rate
parser.add_argument('--save_every_n_epochs', type=float, default=0.001) # save model every n epochs
parser.add_argument('--seed', type=str, default=222) # seed
parser.add_argument('--num_workers', type=str, default=2) # num_workers, 基本2の倍数が望ましい
parser.add_argument('--output_dim', type=int, default=10)


# # config file (yaml)
# # === I/O ===
# device: null
# exp_name: null
# # === Data ===
# num_workers: 4
# pin_memory: True
# # === Model ===
# output_dim: 10
# num_blocks: 1
# # === Training ===
# batch_size: 32
# epochs: 50
# lr: 0.01
# accum_grad: 1
# clip_grad: 1.0
# save_model_every: 10
# patience: 5
# early_stop_mode: 'min'
# # === Logging ===
# log_every: 2





args = parser.parse_args() # Namespace object

# argsをconfigに変換
cfg = vars(args)

# seedの固定
utils.fix_seed(seed=args.seed, fix_gpu=False) # for seed control

# setup
now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
cfg["outdir"] = args.workdir + '/results/' + now # for output
if not os.path.exists(cfg["outdir"]):
    os.makedirs(cfg["outdir"])
cfg["device"] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device


def main():
    if args.train:
        # training mode
        start = time.time() # for time stamp
        # 1. data prep
        train_loader, test_loader = prepare_data()
        cfg["num_training_data"] = len(train_loader)
        cfg["num_test_data"] = len(test_loader)        
        # 2. model prep
        model, criterion, optimizer, scheduler = prepare_model()
        # 3. training
        model, train_loss, test_loss, accuracies = fit(
            model, train_loader, test_loader, criterion, optimizer, scheduler
            )
        # 4. modify config
        components = utils.get_component_list(model, optimizer, criterion, cfg["device"], scheduler)
        cfg.update(components) # update config
        elapsed_time = utils.timer(start) # for time stamp
        cfg["elapsed_time"] = elapsed_time
        # 5. save experiment & config
        utils.save_experiment(
            experiment_name=now, config=cfg, model=model, train_losses=train_loss,
            test_losses=test_loss, accuracies=accuracies, classes=None, base_dir=cfg["outdir"]
            )
        print(">> Done training")
    else:
        # inference mode
        # データ読み込みをtestのみに変更などが必要
        pass


if __name__ == '__main__':
    main()