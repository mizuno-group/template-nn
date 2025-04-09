# -*- coding: utf-8 -*-
"""

notebook上での使用などインタラクティブなクラスを提供

ToDo
- testする.

@author: tadahaya
"""
# packages installed in the current environment
import os, time, yaml, inspect
from typing import Optional, Union, List, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from schedulefree import RAdamScheduleFree

# original packages in src
from .src import utils
from .src import data_handler as dh
from .src.trainer import Trainer
from .src.models import MyNet

# 抽象クラス
class BaseInteractive:
    def __init__(self):
        pass

    def init_model(self):
        """ initialize model """
        raise NotImplementedError

    def set_params(self):
        """ set parameters """
        raise NotImplementedError

    def prep_data(self):
        """ prepare data """
        raise NotImplementedError

    def fit(self):
        """ train the model """
        raise NotImplementedError

    def predict(self):
        """ conduct prediction """
        raise NotImplementedError

    def get_latent(self):
        """ get latent representation """
        raise NotImplementedError

    def load_model(self):
        """ load model """
        raise NotImplementedError


class Interactive(BaseInteractive):
    """ class for training and prediction """
    def __init__(
            self,
            config_path: str=None,
            train_data=None,
            test_data=None,
            train_label=None,
            test_label=None,
            outdir: Optional[str]=None,
            exp_name: Optional[str]=None,
            seed: int=42,
            **kwargs
            ):
        # arguments
        assert config_path is not None, "!! Give config_path !!"
        assert outdir is not None, "!! Give outdir !!"
        self.train_data, self.test_data = train_data, test_data
        self.train_label, self.test_label = train_label, test_label
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.config_path = config_path
        if exp_name is not None:
            self.config["exp_name"] = exp_name
        self.outdir = outdir
        # initialize
        self.train_dataset = None
        self.test_dataset = None
        self.model = None
        self.trainer = None
        # fix seed
        g, seed_worker = utils.fix_seed(seed, fix_cuda=False)
        self._seed = {"seed": seed, "g": g, "seed_worker": seed_worker}
        # prepare model
        self.init_model()
        self.set_params(**kwargs)


    def init_model(self):
        """
        prepare model
        hard coded parameters
        
        """
        self.model = MyNet()
        for param in self.model.parameters(): # 呼び出し方によってたまに外れる恐れがあるため明示
            param.requires_grad = True
        optimizer = RAdamScheduleFree(self.model.parameters(), lr=float(self.config["lr"]), betas=(0.9, 0.999))
        loss_fn = nn.CrossEntropyLoss() # hard coded
        self.trainer = Trainer(
            self.config, self.model, optimizer, loss_fn, outdir=self.outdir
            )


    def set_params(self, **kwargs):
        """
        set parameters
                
        """
        # set parameters
        if kwargs is not None:
            for k, v in kwargs.items():
                if k in self.config:
                    self.config[k] = v


    def prep_data(self):
        """ prepare data """
        self.train_dataset = dh.MyDataset(data=self.train_data, label=self.train_label)
        train_loader = dh.prep_dataloader(
            dataset=self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"],
            g=self._seed["g"]
            seed_workers=self._seed["seed_worker"]
            )
        if self.data_test is None:
            return train_loader, None
        else:
            self.test_dataset = dh.MyDataset(data=self.test_data, label=self.test_label)
            test_loader = dh.prep_dataloader(
                dataset=self.test_dataset,
                batch_size=self.config["batch_size"],
                shuffle=False,
                num_workers=self.config["num_workers"],
                pin_memory=self.config["pin_memory"],
                g=self._seed["g"],
                seed_worker=self._seed["seed_worker"]
                )
            return train_loader, test_loader


    def fit(self, train_loader, test_loader, callbacks:list=None, verbose:bool=True):
        """
        train the model
        
        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            training data loader
        
        test_loader: torch.utils.data.DataLoader
            test data loader

        callbacks: list
            list of callback functions to be called during training

        verbose: bool
            whether to print the training progress
        
        """
        if callbacks is not None:
            self.trainer.set_callbacks(callbacks)
        self.trainer.train(train_loader, test_loader)
        if verbose:
            print(">> Training is done.")


    def predict(self, data_loader=None):
        """ prediction """
        if data_loader is None:
            raise ValueError("!! Give data_loader !!")
        if self.model is None:
            raise ValueError("!! fit or load_model first !!")
        self.model.eval()
        preds = []
        probs = []
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                output = self.model(data)
                preds.append(output.argmax(dim=1).cpu().numpy())
                probs.append(output.cpu().numpy())
        return np.concatenate(preds), np.concatenate(probs)


    def get_latent(self, dataset=None, indices:list=[]):
        """
        get latent representation
        
        """
        if dataset is None:
            dataset = self.train_dataset
        if self.model is None:
            raise ValueError("!! fit or load_model first !!")
        self.model.eval()
        num_data = len(dataset)
        if len(indices) == 0:
            indices = list(range(num_data))
        reps = []
        with torch.no_grad():
            for i in indices:
                data, _ = dataset[i]
                data = (data.to(self.device).unsqueeze(0) for x in data) # add batch dimension
                output = self.model(data)
                reps.append(output.cpu().numpy().reshape(1, -1))  # del batch dimension
        return np.vstack(reps)


    def load_model(self, model_path: str, config_path: str=None):
        """ load the model """
        if config_path is not None:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        # initialize the model
        model_params = inspect.signature(MyNet.__init__).parameters
        model_args = {k: self.config[k] for k in model_params if k in self.config}
        self.model = MyNet(**model_args)
        self.model.load_state_dict(torch.load(model_path))