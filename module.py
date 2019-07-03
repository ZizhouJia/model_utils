import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class Module(nn.Module):
    def __init__(self):
        super(Module,self).__init__()
        self.model_save_path="checkpoints"
        self._paral=False
    
    def switch_save_path(self,model_save_path):
        if(not os.path.exists(model_save_path)):
            print("Warning: The path doesn't exist, Fail to switch the path to "+str(model_save_path))
            return
        else:
            self.model_save_path=model_save_path

    
    def save_params(self,file_name):
        if(not os.path.exists(self.model_save_path)):
            os.makedirs(file_name)
        save_path=os.path.join(self.model_save_path,file_name)
        if(self._paral):
            torch.save(self.module.state_dict(),save_path)
        else:
            torch.save(self.state_dict(),save_path)

    def load_params(self,file_name):
        save_path=os.path.join(self.model_save_path,file_name)
        if(not os.path.exists(save_path)):
            print("Warning: Path not exists, Fail to load params")
            return 

        if(not self._paral):
            self.load_state_dict(torch.load(save_path))
        else:
            model_state=torch.load(save_path)
            new_state=OrderedDict()
            for key,v in model_state.items():
                name="module."+key
                new_state[name]=v
            self.load_state_dict(new_state)


            
        

