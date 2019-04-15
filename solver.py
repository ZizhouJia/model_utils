import os
from datetime import datetime
from tensorboardX import SummaryWriter
import torch
import numpy as np
import torch.nn as nn
import torchvision.utils as vutils

from . import draw
from . import config as cfg

def get_time_string():
    dt = datetime.now()
    return dt.strftime("%Y%m%d%H%M")

class request:
    def __init__(self):
        self.epoch=0
        self.iteration=0
        self.step=0
        self.data=None

class solver(object):
    def __init__(self):
        self.models = None
        self.task_name = None
        self.writer = None
        self.optimizers = None
        self.config=None
        self.save_path = "checkpoints"
        self.timestemp = get_time_string()
        # self.kernel_processer.set_models(models)
        # self.kernel_processer.set_optimizers(optimizers)

    def set_config(self,config):
        self.config=config

    def get_config(self):
        return self.config

    def set_task_name(self,task_name):
        self.task_name=task_name

    def set_models(self,models):
        self.models=models

    def set_optimizers(self,optimizers):
        self.optimizers=optimizers

    def set_timestemp(self,timestemp):
        self.timestemp=timestemp

    def init_summary_writer(self):
        if(self.task_name is None):
            self.writer=SummaryWriter("runs/unkonwn-"+self.timestemp)
        else:
            self.writer=SummaryWriter("runs/"+self.task_name+"-"+self.timestemp)


    def get_models(self):
        return self.models

    def init_models(self, init_func):
        for model in self.models:
            init_func(model)

    def set_optimizers(self, optimizers):
        self.optimizers = optimizers

    def train_mode(self):
        for model in self.models:
            model.train()

    def eval_mode(self):
        for model in self.models:
            model.eval()

    def zero_grad_for_all(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def write_log(self, loss, index):
        for key in loss:
            self.writer.add_scalar("scalar/"+key, loss[key], index)

    def print_log(self, loss, epoch, iteration):
        print("in epoch %d iteration %d " % (epoch, iteration))
        print(loss)

    def write_log_image(self, image, index):
        for key in image:
            self.writer.add_image(
                'image/'+key, vutils.make_grid(image[key], 1), index)

    def save_params(self, epoch=-1):
        path = self.save_path
        if(not os.path.exists(path)):
            os.mkdir(path)

        path = os.path.join(path, self.task_name)
        if(not os.path.exists(path)):
            os.mkdir(path)

        path = os.path.join(path, self.timestemp)
        if(not os.path.exists(path)):
            os.mkdir(path)

        if(epoch != -1):
            path = os.path.join(path, str(epoch))
            if(not os.path.exists(path)):
                os.mkdir(path)

        file_name = "model"
        for i in range(0, len(self.models)):
            torch.save(self.models[i].state_dict(), os.path.join(
                path, file_name+"-"+str(i)+".pkl"))

        print("the models params "+self.task_name +
              " has already been saved "+self.timestemp)

    def restore_params(self, task_name, time_string, epoch=-1):
        path = self.save_path
        path = os.path.join(path, task_name)
        path = os.path.join(path, time_string)
        if(epoch != -1):
            path = os.path.join(path, str(epoch))
        file_name = "model"

        for i in range(0, len(self.models)):
            self.models[i].load_state_dict(torch.load(
                os.path.join(path, file_name+"-"+str(i)+".pkl")))

        print("the models params "+self.task_name +
              " has already been restored "+self.timestemp)



    def save_models(self, epoch=-1):
        path = self.save_path
        if(not os.path.exists(path)):
            os.mkdir(path)

        path = os.path.join(path, self.task_name)
        if(not os.path.exists(path)):
            os.mkdir(path)

        path = os.path.join(path, self.timestemp)
        if(not os.path.exists(path)):
            os.mkdir(path)

        if(epoch != -1):
            path = os.path.join(path, str(epoch))
            if(not os.path.exists(path)):
                os.mkdir(path)

        file_name = "model"
        for i in range(0, len(self.models)):
            torch.save(self.models[i], os.path.join(
                path, file_name+"-"+str(i)+".pkl"))

        print("the models "+self.task_name +
              " has already been saved "+self.timestemp)

    def restore_models(self,task_name,time_string, epoch=-1):
        path = self.save_path
        path = os.path.join(path, task_name)
        path = os.path.join(path, time_string)
        if(epoch != -1):
            path = os.path.join(path, epoch)
        file_name = "model"

        self.models = []
        i = 0
        while(True):
            current_file = os.path.join(path, file_name+"-"+str(i)+".pkl")
            if(not os.path.exists(current_file)):
                break;
            self.models.append(torch.load(current_file))
            i += 1

        print("the models params "+self.task_name + \
              " has already been restored "+self.timestemp)

    def restore_params_with_path(self,path):
        file_name="model"
        for i in range(0, len(self.models)):
            self.models[i].load_state_dict(torch.load(
                os.path.join(path, file_name+"-"+str(i)+".pkl")))
        print("the models params has already been restored")


#solver that pair with the kernel processer
class common_solver(solver):
    def __init__(self):
        super(common_solver,self).__init__()
        self.request=request()

    def load_config(self):
        if(self.config is None):
            raise NotImplementedError("the main_loop begin before the config set")
        self.epochs=self.config["epochs"]
        self.learning_rate_decay_epochs=self.config["learning_rate_decay_epochs"]
        self.train_loader,self.validate_loader,self.test_loader=self.config["dataset_function"](**self.config["dataset_function_params"])
        self.if_train=self.config["train"]
        self.if_validate=self.config["validate"]
        self.if_test=self.config["test"]
        self.begin_epoch=self.config["begin_epoch"]
        self.restored_path=self.config["restored_path"]
        if(self.restored_path is not None):
            self.restore_params_with_path(self.restored_path)

    def main_loop(self):
        self.load_config()
        #the init process default is nothing to do
        self.init()
        #the whole training process
        if(self.if_train):
            for epoch in range(self.begin_epoch,self.epochs):
                self.request.epoch=epoch
                self.train_mode()
                self.before_train()
                for step,data in enumerate(self.train_loader):
                    self.request.step=step
                    self.request.data=data
                    self.train()
                    self.request.iteration+=1
                self.after_train()
                if(self.if_validate):
                    #validate process
                    self.eval_mode()
                    self.before_validate()
                    for step,data in enumerate(self.validate_loader):
                        self.request.step=step
                        self.request.data=data
                        self.validate()
                    self.after_validate()
        if(self.if_test):
            #the test process
            self.eval_mode()
            self.before_test()
            for step,data in enumerate(self.test_loader):
                self.request.step=step
                self.request.data=data
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

class common_classify_solver(common_solver):
    def __init__(self):
        super(common_classify_solver,self).__init__()
        self.loss_function=nn.CrossEntropyLoss()
        self.pred=[]
        self.label=[]
        self.image_name=[]
        self.loss_value=[]
        self.element_number=[]
        self.best_acc=0.0

    def empty_list(self):
        self.pred=[]
        self.label=[]
        self.image_name=[]
        self.loss_value=[]
        self.element_number=[]

    def before_test(self):
        if(self.if_train):
            self.restore_params(self.task_name,self.timestemp,"best")

    def before_train(self):
        if(self.request.epoch in self.learning_rate_decay_epochs):
            for optimizer in self.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr']=param_group['lr']*0.1

    def train(self):
        model=self.models[0]
        optimizer=self.optimizers[0]
        x,y,_=self.request.data
        x=x.cuda()
        y=y.cuda()
        pred=model(x)
        loss=self.loss_function(pred,y)
        loss.backward()
        optimizer.step()
        self.zero_grad_for_all()
        if(self.request.iteration%20==0):
            show_dict={}
            pred_label=torch.max(pred,1)[1]
            acc=torch.sum((pred_label==y).float())/x.size(0)
            show_dict["train_acc"]=acc.detach().cpu().item()
            show_dict["train_loss"]=loss.detach().cpu().item()
            self.write_log(show_dict,self.request.iteration)
            self.print_log(show_dict,self.request.epoch,self.request.step)

    def validate(self):
        model=self.models[0]
        x,y,image_name=self.request.data
        x=x.cuda()
        y=y.cuda()
        pred=model(x)
        loss=self.loss_function(pred,y)
        pred=pred.detach().cpu().numpy()
        y=y.detach().cpu().numpy()
        for i in range(0,pred.shape[0]):
            self.pred.append(pred[i,:])
            self.label.append(y[i])
            self.image_name.append(image_name[i])
        self.loss_value.append(loss.detach().cpu().item())
        self.element_number.append(x.size(0))

    def test(self):
        self.validate()

    def after_validate(self):
        preds=np.array(self.pred)
        labels=np.array(self.label)
        pred_label=np.argmax(preds,axis=1)
        acc=np.sum((pred_label==labels).astype(np.float))/preds.shape[0]
        total_loss=0
        total_emement=0
        for i in range(0,len(self.loss_value)):
            total_loss+=(self.loss_value[i]*self.element_number[i])
            total_emement+=self.element_number[i]
        loss=total_loss/total_emement
        write_dict={}
        write_dict["validate_loss"]=loss
        write_dict["validate_acc"]=acc
        self.write_log(write_dict,self.request.epoch)
        self.print_log(write_dict,self.request.epoch,0)
        self.empty_list()
        if(write_dict["validate_acc"]>self.best_acc):
            self.best_acc=write_dict["validate_acc"]
            self.save_params("best")

    def after_test(self):
        preds=np.array(self.pred)
        labels=np.array(self.label)
        pred_label=np.argmax(preds,axis=1)
        acc=np.sum((pred_label==labels).astype(np.float))/preds.shape[0]
        total_loss=0
        total_emement=0
        for i in range(0,len(self.loss_value)):
            total_loss+=self.loss_value[i]*self.element_number[i]
            total_emement+=self.element_number[i]
        loss=total_loss/total_emement
        confusion_matrix=np.zeros((preds.shape[1],preds.shape[1]))
        for i in range(0,len(pred_label)):
            confusion_matrix[labels[i],pred_label[i]]+=1

        write_dict={}
        write_dict["test_loss"]=loss
        write_dict["test_acc"]=acc
        self.write_log(write_dict,0)
        self.print_log(write_dict,self.request.epoch,0)
        draw.plotCM(confusion_matrix,"confusion_matrix.jpg")
        f=open("pred.txt","w")
        for i in range(0,len(self.image_name)):
            f.write(self.image_name[i]+" "+str(pred_label[i])+" \n")
        f.close()
        self.empty_list()

class t_sne_solver(common_solver):
    def __init__(self):
        super(t_sne_solver,self).__init__()
        self.features=[]
        self.labels=[]

    def main_loop(self):
        self.load_config()
        model=self.models[0]
        for step,data in enumerate(self.test_loader):
            x,y,_=data
            x=x.cuda()
            y=y.cuda()
            print(step)
            feature=model(x)
            feature=feature.detach().cpu().numpy()
            y=y.detach().cpu().numpy()
            for i in range(0,feature.shape[0]):
                self.features.append(feature[i])
                self.labels.append(y[i])
        #begin t-SNE
        draw.plotTSNE(np.array(self.features),np.array(self.labels),"t_SNE.jpg")
