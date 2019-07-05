# -*- coding: utf-8 -*-
import os
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils as vutils

# from . import draw
from . import saver
from . import writer


def get_time_string():
    dt = datetime.now()
    return dt.strftime("%Y%m%d%H%M")


class request:
    def __init__(self):
        self.epoch = 0
        self.iteration = 0
        self.step = 0
        self.data = None


class base_config(object):
    def __init__(self):
        self.task_name = "defualt"
        self.model_classes = []
        self.model_params = []
        self.optimizer_function = None
        self.optimizer_params = {}
        # set the device automatic if the device_use is None, else set the certaion numbers
        self.device_use = None
        self.memory_use = None  # set the memory use
        self.writer_open = True  # open the summart writer
        self.timestemp = None  # set none to generate a new stemp
        self.model_save_path = "checkpoints"
        self.logger_save_path="logs"
        self.image_save_path="images"
        self.summary_writer_save_path="runs"


class solver(object):
    def __init__(self):
        self.models = None
        self.task_name = None
        self.writer = None
        self.optimizers = None
        self.config = None
        self.timestemp = None
        self.writer = None
        self.saver = None

    @staticmethod
    def get_default_config():
        default_config = base_config()
        return default_config

    def load_config(self):
        # set task_name
        if(self.config.task_name is not None):
            self.task_name = self.config.task_name
        else:
            self.task_name = "default"

        # set timestemp
        if(self.config.timestemp is None):
            self.timestemp = get_time_string()
        else:
            self.timestemp = str(self.timestemp)

        # set summary writer if the writer close, it is none
        self.writer = writer.writer(self.task_name+"-"+self.timestemp,
                summary_writer_default_path=self.config.summary_writer_save_path,
                logger_default_path=self.config.logger_save_path,
                image_default_path=self.config.image_save_path
                ,output_log=self.config.writer_open)

        # set models
        models = []
        torch.cuda.set_device(self.config.device_use[0])
        for i in range(0, len(self.config.model_classes)):
            model_class = self.config.model_classes[i]
            params = self.config.model_params[i]
            models.append(model_class(**params))
        self.set_models(models)
        self.parallel(self.config.device_use)

        # set optimizer
        if(self.config.optimizer_function is not None):
            optimizers = self.config.optimizer_function(
                self.models, **self.config.optimizer_params)
            self.set_optimizers(optimizers)

        # set saver
        self.saver = saver.saver(
            self.config.model_save_path, (len(self.config.device_use) > 1))

    def get_task_identifier(self):
        return self.task_name+"-"+self.timestemp

    def set_models(self, models):
        self.models = models

    def get_models(self):
        return self.models

    def set_optimizers(self, optimizers):
        self.optimizers = optimizers

    def get_optimizers(self):
        return self.optimizers

    def set_config(self, config):
        self.config = config
        self.load_config()

    def get_config(self):
        return self.config

    def init_models(self, init_func):
        for model in self.models:
            init_func(model)

    def train_mode(self):
        for model in self.models:
            model.train()

    def eval_mode(self):
        for model in self.models:
            model.eval()

    def optimize_all(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad_for_all(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    # parallel function to send the models to certain device

    def parallel(self, device_ids=[0]):
        print("use device: "+str(device_ids))
        # set gpu mode
        for i in range(0, len(self.models)):
            self.models[i] = self.models[i].cuda()
        # sigle gpu
        if(len(device_ids) == 1):
            return self.models
        # multiple GPU
        ret = []
        for i in range(0, len(self.models)):
            ret.append(nn.DataParallel(self.models[i], device_ids=device_ids))
        self.models = ret
        # set the parallal flag
        for model in self.models:
            model._paral = True
        return ret


# solver that pair with the kernel processer
class common_solver(solver):
    def __init__(self):
        super(common_solver, self).__init__()
        self.request = request()

    @staticmethod
    def get_default_config():
        config = solver.get_default_config()
        config.epochs = 0
        config.learning_rate_decay_epochs = []
        config.train = True
        config.validate = True
        config.test = True
        config.begin_epoch = 0
        config.dataset_function = None
        config.dataset_function_params = {}
        return config

    def load_config(self):
        super(common_solver, self).load_config()
        if(self.config is None):
            raise NotImplementedError(
                "the main_loop begin before the config set")
        self.epochs = self.config.epochs
        self.train_loader, self.validate_loader, self.test_loader = self.config.dataset_function(
            **self.config.dataset_function_params)
        self.if_train = self.config.train
        self.if_validate = self.config.validate
        self.if_test = self.config.test
        self.begin_epoch = self.config.begin_epoch

    def main_loop(self):
        # the init process default is nothing to do
        self.init()
        # the whole training process
        if(self.if_train):
            for epoch in range(self.begin_epoch, self.epochs):
                self.request.epoch = epoch
                self.train_mode()
                self.before_train()
                for step, data in enumerate(self.train_loader):
                    self.request.step = step
                    self.request.data = data
                    self.train()
                    self.request.iteration += 1
                self.after_train()
                if(self.if_validate):
                    # validate process
                    self.eval_mode()
                    self.before_validate()
                    for step, data in enumerate(self.validate_loader):
                        self.request.step = step
                        self.request.data = data
                        self.validate()
                    self.after_validate()
        if(self.if_test):
            # the test process
            self.eval_mode()
            self.before_test()
            for step, data in enumerate(self.test_loader):
                self.request.step = step
                self.request.data = data
                self.test()
            self.after_test()

    def train(self):
        pass

    def test(self):
        pass

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_test(self):
        pass

    def after_test(self):
        pass

    def before_validate(self):
        pass

    def after_validate(self):
        pass

    def validate(self):
        self.test()

    def init(self):
        pass

    def end(self):
        pass
