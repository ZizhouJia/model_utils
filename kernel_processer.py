import torch
import torch.nn as nn

class kernel_processer(object):
    def __init__(self):
        self.models=None
        self.optimizers=None

    def set_models(self,models):
        self.models=models

    def set_optimizers(self,optimizers):
        self.optimizers=optimizers

    def update_optimizers(self,epoch,step,total_data_numbers):
        pass

    def on_finish(self):
        pass

    def zero_grad_for_all(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def train(self,step,data):
        raise NotImplementedError

    def test(self,step,data):
        raise NotImplementedError

    def evaluate(self,step,data):
        raise NotImplementedError

    def update_optimizers(self,epoch,step,total_data_numbers):
        pass
