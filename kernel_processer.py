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

    def tencrop_process(self,data):
        x=data[0]
        y=data[1]
        if(len(x.size())==5):
            repeat_size=x.size(1)
            y_list=[]
            for i in range(0,repeat_size):
                y_list.append(y)
            y=torch.stack(y_list,0)
            new_size=list(y.size())
            temp=new_size[0]
            value=list(range(0,len(new_size)))
            value[0]=1
            value[1]=0
            y=y.permute(value)
            new_size[1]=new_size[0]*new_size[1]
            del new_size[0]
            y=y.contiguous().view(new_size)
            x=x.view(-1,x.size(2),x.size(3),x.size(4))
        return (x,y)




    def zero_grad_for_all(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def train(self,step,data):
        raise NotImplementedError

    def test(self,step,data):
        raise NotImplementedError

    def update_optimizers(self,epoch,step,total_data_numbers):
        pass
