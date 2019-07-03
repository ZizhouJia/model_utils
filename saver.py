import torch
import os
from collections import OrderedDict


class saver(object):
    def __init__(self, model_save_path, paral):
        self.model_save_path = model_save_path
        self._paral = paral

    def switch_save_path(self, model_save_path):
        if(not os.path.exists(model_save_path)):
            print(
                "Warning: The path doesn't exist, Fail to switch the path to "+str(model_save_path))
            return
        else:
            self.model_save_path = model_save_path

    def save_params(self, model, file_name):
        if(not os.path.exists(self.model_save_path)):
            os.makedirs(self.model_save_path)
        save_path = os.path.join(self.model_save_path, file_name)
        if(self._paral):
            torch.save(model.module.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)
        print("the model has already been saved")

    def load_params(self, model, file_name):
        save_path = os.path.join(self.model_save_path, file_name)
        if(not os.path.exists(save_path)):
            print("Warning: Path not exists, Fail to load params")
            return

        if(not self._paral):
            model.load_state_dict(torch.load(save_path))
        else:
            model_state = torch.load(save_path)
            new_state = OrderedDict()
            for key, v in model_state.items():
                name = "module."+key
                new_state[name] = v
            model.load_state_dict(new_state)
        print("the model has already been loaded")
