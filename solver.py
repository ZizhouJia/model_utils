# -*- coding: utf-8 -*-
import os
from collections import OrderedDict
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

# from . import draw


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
        self.model_classs = []
        self.model_params = []
        self.optimizer_function = None
        self.optimizer_params = {}
        # set the device automatic if the device_use is None, else set the certaion numbers
        self.device_use = None
        self.mem_use = None  # set the memory use
        self.summary_writer_open = True  # open the summart writer
        self.model_save_path = "checkpoints"
        self.timestemp = None  # set none to generate a new stemp


class solver(object):
    def __init__(self):
        self.models = None
        self.task_name = None
        self.writer = None
        self.optimizers = None
        self.config = None
        self.model_save_path = None
        self.timestemp = None
        self.writer = None

    def get_defualt_config():
        default_config = base_config()
        return default_config

    def load_config(self):
        # set task_name
        if(self.config.task_name is None):
            self.task_name = self.config.task_name
        else:
            self.task_name = "default"

        # set timestemp
        if(self.config.timestemp is None):
            self.timestemp = get_time_string()
        else:
            self.timestemp = str(self.timestemp)

        # set model_save_path
        if(self.config.model_save_path is None):
            self.model_save_path = "checkpoints"
        else:
            self.model_save_path = self.config.model_save_path

        # set summary writer if the writer close, it is none
        if(self.config.summary_writer_open):
            self.writer = SummaryWriter(
                "runs/"+self.task_name+"-"+self.timestemp)

        # set models
        models = []
        torch.cuda.set_device(self.config.device_use[0])
        for i in range(0, len(self.config.model_class)):
            model_class = self.config.model_class[i]
            param = self.config.model_params[i]
            models.append(model_class(**param))
        self.set_models(models)
        self.parallel(self.config.device_use)

        # set optimizer
        if(self.config["optimizer_function"] is not None):
            optimizers = self.config["optimizer_function"](
                self.models, **self.config["optimizer_params"])
            self.set_optimizers(optimizers)

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

    def write_log(self, log_dict, index):
        if(self.writer is None):
            print("Warning:the writer is forbidon")
            return
        for key in log_dict:
            self.writer.add_scalar("scalar/"+key, log_dict[key], index)

    def print_log(self, log_dict):
        print(log_dict)

    def write_log_image(self, image_dict, index):
        if(self.writer is None):
            print("Warning the writer is forbidon")
            return
        for key in image_dict:
            self.writer.add_image(
                'image/'+key, vutils.make_grid(image_dict[key], 1), index)

    def save_single_model(self, model, model_path):
        if(len(self.config.device_use) > 1):
            torch.save(model.module, model_path)
        else:
            torch.save(model, model_path)

    def restore_single_model(self, model_path):
        model = torch.load(model_path)
        model = model.cuda()
        if(len(self.config.device_use) > 1):
            model = nn.DataParallel(model, device_ids=self.config.device_use)
        return model

    def save_single_model_params(self, model, model_path):
        if(len(self.config.device_use) > 1):
            torch.save(model.module.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)

    def restore_single_model_params(self, model, model_path):
        if(len(self.config.device_use) <= 1):
            model.load_state_dict(torch.load(model_path))
        else:
            model_state = torch.load(model_path)
            new_state = OrderedDict()
            for key, v in model_state.items():
                name = "module."+key
                new_state[name] = v
            model.load_state_dict(new_state)
        return model

    def save_params(self, save_string=None, with_timestemp=True):
        path = self.model_save_path
        if(not os.path.exists(path)):
            os.mkdir(path)

        path = os.path.join(path, self.task_name)
        if(not os.path.exists(path)):
            os.mkdir(path)

        if(with_timestemp):
            path = os.path.join(path, self.timestemp)
            if(not os.path.exists(path)):
                os.mkdir(path)

        if(save_string is not None):
            path = os.path.join(path, str(save_string))

        if(not os.path.exists(path)):
            os.mkdir(path)

        self.save_params_with_path(path)

    def restore_params(self, save_string=None, task_name=None, timestemp=None):
        path = self.model_save_path
        path = os.path.join(path, task_name)

        if(task_name is None):
            path = os.path.join(path, self.task_name)
        else:
            path = os.path.join(path, str(task_name))

        if(timestemp is None):
            path = os.path.join(path, self.timestemp)
        else:
            path = os.path.join(path, str(timestemp))

        if(save_string is not None):
            path = os.path.join(path, str(save_string))

        self.restore_params_with_path(path)

    def save_models(self, save_string=None, with_timestemp=True):
        path = self.model_save_path
        if(not os.path.exists(path)):
            os.mkdir(path)

        path = os.path.join(path, self.task_name)
        if(not os.path.exists(path)):
            os.mkdir(path)

        if(with_timestemp):
            path = os.path.join(path, self.timestemp)
            if(not os.path.exists(path)):
                os.mkdir(path)

        if(save_string is not None):
            path = os.path.join(path, str(save_string))

        if(not os.path.exists(path)):
            os.mkdir(path)
        self.save_models_with_path(path)

    def restore_models(self, save_string=None, task_name=None, timestemp=None):
        path = self.model_save_path
        path = os.path.join(path, task_name)

        if(task_name is None):
            path = os.path.join(path, self.task_name)
        else:
            path = os.path.join(path, str(task_name))

        if(timestemp is None):
            path = os.path.join(path, self.timestemp)
        else:
            path = os.path.join(path, str(timestemp))

        if(save_string is not None):
            path = os.path.join(path, str(save_string))

        self.restore_models_with_path(path)

    def save_models_with_path(self, path):
        file_name = "model"
        for i in range(0, len(self.models)):
            model_path = os.path.join(path, file_name+"_"+str(i)+",pkl")
            self.save_single_model(self.models[i], model_path)

        print("the models "+self.task_name +
              " has already been saved in :"+str(path))

    def restore_models_with_path(self, path):
        file_name = "model"
        self.models = []
        i = 0
        while(True):
            current_file = os.path.join(path, file_name+"_"+str(i)+".pkl")
            if(not os.path.exists(current_file)):
                break
            self.models.append(torch.load(current_file))
            i += 1
        self.parallel(self, self.config.device_use)

        print("the models "+self.task_name +
              " has already been restored from "+str(path))

    def restore_params_with_path(self, path):
        file_name = "model"
        for i in range(0, len(self.models)):
            model_path = os.path.join(path, file_name+"_"+str(i)+",pkl")
            self.restore_single_model_params(self.models[i], model_path)

        print("the models params "+self.task_name +
              " has already been restored from "+str(path))

    def save_params_with_path(self, path):
        file_name = "model"
        for i in range(0, len(self.models)):
            model_path = os.path.join(path, file_name+"_"+str(i)+",pkl")
            self.save_single_model_params(self.models[i], model_path)

        print("the models params "+self.task_name +
              " has already been saved in :"+str(path))

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
        return ret


# solver that pair with the kernel processer
class common_solver(solver):
    def __init__(self):
        super(common_solver, self).__init__()
        self.request = request()

    def get_defualt_config():
        config = solver.get_defualt_config()
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
