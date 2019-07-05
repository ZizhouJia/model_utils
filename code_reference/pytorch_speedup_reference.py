import torch

#speed up the static tensor map
torch.backends.cudnn.benchmark=True

#deterministic convolutions for reimplementated experiment
torch.backends.cudnn.deterministic=True

#free the unused memory
torch.cuda.empty_cache()

#set pin memory
pin_memory=True

#improve the speed of the dataloader



