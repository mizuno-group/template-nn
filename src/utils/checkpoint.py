# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

checkpointや学習履歴を保存・読み込みする関数をまとめる. 

@author: tadahaya
"""
from pathlib import Path
from typing import Any, Dict, Tuple
import os
import json
import yaml

import torch


def save_experiment(model, optimizer, scheduler, history:Dict, save_dir:str, file_name:str) -> None:
    """
    save the experiment: config, model, metrics, and progress plot
    
    outdir
    ├── experiment_name (resdir)
        ├── config.yaml
        ├── history.json
        ├── progress_loss.tif
        ├── model_final.pt
        ├── model_1.pt
        ├── model_2.pt
        ├── ...
    
    """
    os.makedirs(outdir, exist_ok=True)
    # save config
    configfile = os.path.join(outdir, 'config.yaml')
    with open(configfile, 'w') as f:
        yaml.dump(config, f, default_flow_style=False) 
    # save history
    historyfile = os.path.join(outdir, 'history.json')
    with open(historyfile, 'w') as f:
        json.dump(history, f, sort_keys=True, indent=4)
    # save the model
    save_checkpoint(model=model, optimizer=optimizer, name="final", outdir=outdir)


def save_checkpoint(model, optimizer, name, outdir):
    """
    save the model checkpoint
    
    """
    cpfile = os.path.join(outdir, f"model_{name}.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        cpfile
    )


def load_experiments(model, optimizer, resdir, checkpoint_name="model_final"):
    """
    load the experiment

    Parameters
    ----------
    model: nn.Module
        initialized model

    optimizer: torch.optim
        initialized optimizer

    resdir: str
        the result directory
    
    checkpoint_name: str
        the checkpoint name, like model_final
    
    """
    # load config
    configfile = os.path.join(resdir, "config.yaml")
    with open(configfile, 'r') as f:
        config = yaml.safe_load(f)
    # load history
    historyfile = os.path.join(resdir, 'hisotry.json')
    with open(historyfile, 'r') as f:
        history = json.load(f)
    # load model
    pkg = torch.load(os.path.join(resdir, checkpoint_name))
    model.load_state_dict(pkg["model"])
    optimizer.load_state_dict(pkg["optimizer"])
    return model, optimizer, config, history