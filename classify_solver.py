import solver
import torch
import torch.nn as nn


class simple_classify_solver(sovler.sovler):
    def __init__(self, models, model_name, save_path='checkpoints'):
        super(simple_classify_solver, self).__init__(
            models, model_name, save_path)

    def
