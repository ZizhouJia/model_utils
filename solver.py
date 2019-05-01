import os
from datetime import datetime
from tensorboardX import SummaryWriter
import torch
import numpy as np
import torch.nn as nn
import torchvision.utils as vutils
import torchvision
import h5py

from . import draw
from . import config as cfg
from . import module
from . import utils

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
        self.load_config()

    def load_config(self):

        #set task_name
        solver.set_task_name(self.config["task_name"])

        #set summary writer
        if(self.config["summary_writer_open"]):
            self.init_summary_writer()

        #set models
        models=[]
        for i in range(0,len(self.config["model_class"])):
            model_class=self.config["model_class"][i]
            param=self.config["model_params"][i]
            models.append(model_class(**param))
        models=self.parallel(models,self.config["device_use"])
        solver.set_models(models)

        #set optimizer
        if(self.config["optimizer_function"] is not None):
            optimizers=self.config["optimizer_function"](models,**self.config["optimizer_params"])
            solver.set_optimizers(optimizers)

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

    def optimize_all(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad_for_all(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def write_log(self, loss, index):
        if(self.writer is None):
            print("Warning the writer is forbidon")
            return
        for key in loss:
            self.writer.add_scalar("scalar/"+key, loss[key], index)

    def print_log(self, loss, epoch, iteration):
        print("in epoch %d iteration %d " % (epoch, iteration))
        print(loss)

    def write_log_image(self, image, index):
        if(self.writer is None):
            print("Warning the writer is forbidon")
            return
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


    #parallel function to send the models to certain device
    def parallel(self, models, device_ids=[0]):
        #set device
        torch.cuda.set_device(device_ids[0])
        #set gpu mode
        for i in range(0,len(models)):
            models[i]=models[i].cuda()
        #sigle gpu
        if(len(device_ids)==1):
            return models
        #multiple GPU
        ret = []
        for i in range(0, len(models)):
            ret.append(nn.DataParallel(models[i], device_ids=device_ids))
        return ret


#solver that pair with the kernel processer
class common_solver(solver):
    def __init__(self):
        super(common_solver,self).__init__()
        self.request=request()

    def load_config(self):
        super(self,common_solver).load_config()
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


class vedio_classify_solver(common_solver):
    def __init__(self):
        super(vedio_classify_solver.self).__init__()
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


    def load_config(self):
        super(vedio_classify_solver,self).load_config()
        self.learning_rate_decay_iteration=self.config["learning_rate_decay_iteration"]
        self.grad_plenty=self.config["grad_plenty"]
        self.distilling_mode=self.config["distilling_mode"] #default is False
        if(self.distilling_mode):
            self.distilling_loss=module.distilling_classify_loss(0.5)

    def init(self):
        if(len(self.models)==1):
            self.train_type="frame"
        else:
            self.train_type="vedio"

    def forward(self,x):
        if(self.train_type=="frame"):
            pred=self.models[0](x)
            return pred
        if(self.train_type=="vedio"):
            frames=x.size(1)
            x=x.view(-1,x.size(2),x.size(3),x.size(4))
            features=self.models[0](x)
            features=features.view(-1,frames,features.size(1))
            x=features
            pred=self.models[1](x)
            return pred

    def request_data(self):
        if(self.distilling_mode):
            _,x,y,soft_y=self.request.data
            return x,y,soft_y
        else:
            _,x,y=self.request.data
            return x,y,None

    def train(self):
        x,y,soft_y=self.request_data()
        x=x.cuda()
        y=y.cuda()
        pred=self.forward(x)
        loss=None
        if(soft_y is None):
            loss=self.loss_function(pred,y)
        else:
            loss=self.distilling_loss(pred,y,y_soft)

        loss.backward()
        if(self.grad_plenty!=0):
            nn.utils.clip_grad_norm(self.models[0].parameters(), self.grad_plenty, norm_type=2)
        self.optimize_all()
        self.zero_grad_for_all()
        if(self.request.iteration%10==0):
            show_dict={}
            pred_label=torch.max(pred,1)[1]
            acc=torch.sum((pred_label==y).float())/x.size(0)
            show_dict["train_acc"]=acc.detach().cpu().item()
            show_dict["train_loss"]=loss.detach().cpu().item()
            self.write_log(show_dict,self.request.iteration)
            self.print_log(show_dict,self.request.epoch,self.request.step)

        if(self.learning_rate_decay_iteration!=0):
            if((self.request.iteration+1)%self.learning_rate_decay_iteration==0):
                for optimizer in self.optimizers:
                    for param_group in optimizers:
                        param_group['lr']=param_group['lr']*0.95

    def after_train(self):
        if(self.request.epoch in self.learning_rate_decay_epochs):
            for optimizer in self.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr']=param_group['lr']*0.1


    def validate(self):
        model=self.models[0]
        x,y,soft_y=self.request_data()
        x=x.cuda()
        y=y.cuda()
        pred=self.forward(x)
        oss=None
        if(soft_y is None):
            loss=self.loss_function(pred,y)
        else:
            loss=self.distilling_loss(pred,y,y_soft)
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

        confusion_matrix=np.zeros((preds.shape[1],preds.shape[1]))
        for i in range(0,len(pred_label)):
            confusion_matrix[labels[i],pred_label[i]]+=1
        print(confusion_matrix)
        tfpn_matrix=np.zeros((2,2))
        tfpn_matrix[0,0]=confusion_matrix[0,0]+confusion_matrix[0,1]+confusion_matrix[1,0]+confusion_matrix[1,1]
        tfpn_matrix[1,1]=confusion_matrix[2,2]
        tfpn_matrix[1,0]=confusion_matrix[2,0]+confusion_matrix[2,1]
        tfpn_matrix[0,1]=confusion_matrix[0,2]+confusion_matrix[1,2]
        print(tfpn_matrix)
        write_dict["validate_pression"]=tfpn_matrix[1,1]/(tfpn_matrix[1,1]+tfpn_matrix[0,1])
        write_dict["validate_recall"]=tfpn_matrix[1,1]/(tfpn_matrix[1,1]+tfpn_matrix[1,0])
        write_dict["validate_f_score"]=write_dict["validate_pression"]*write_dict["validate_recall"]/(write_dict["validate_pression"]+write_dict["validate_recall"])
        self.write_log(write_dict,self.request.epoch)
        self.print_log(write_dict,self.request.epoch,0)
        self.empty_list()
        if(write_dict["validate_f_score"]>self.best_acc):
            self.best_acc=write_dict["validate_f_score"]
            self.save_params("best")

    def before_test(self):
        if(self.if_train):
            self.restore_params(self.task_name,self.timestemp,"best")

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
        print(confusion_matrix)

        tfpn_matrix=np.zeros((2,2))
        tfpn_matrix[0,0]=confusion_matrix[0,0]+confusion_matrix[0,1]+confusion_matrix[1,0]+confusion_matrix[1,1]
        tfpn_matrix[1,1]=confusion_matrix[2,2]
        tfpn_matrix[1,0]=confusion_matrix[2,0]+confusion_matrix[2,1]
        tfpn_matrix[0,1]=confusion_matrix[0,2]+confusion_matrix[1,2]
        print(tfpn_matrix)
        write_dict["test_pression"]=tfpn_matrix[1,1]/(tfpn_matrix[1,1]+tfpn_matrix[0,1])
        write_dict["test_recall"]=tfpn_matrix[1,1]/(tfpn_matrix[1,1]+tfpn_matrix[1,0])
        write_dict["test_f_score"]=write_dict["test_pression"]*write_dict["test_recall"]/(write_dict["test_pression"]+write_dict["test_recall"])

        write_dict={}
        write_dict["test_loss"]=loss
        write_dict["test_acc"]=acc
        self.write_log(write_dict,0)
        self.print_log(write_dict,self.request.epoch,0)
        draw.plotCM(confusion_matrix,self.task_name+"_confusion_matrix.jpg")
        draw.plotCM(tfpn_matrix,self.task_name+"_tfpn_matrix.jpg")
        self.empty_list()


class emssemble_solver(vedio_classify_solver):
    def __init__(self):
        self.super(emssemble_solver,self).__init__()
        self.embed_models=[]

    def load_config(self):
        self.if_train=False
        self.if_test=True
        self.model_list=self.config["model_list"]
        self.model_param_list=self.config["model_param_list"]
        self.save_path=self.config["save_path"]
        self.train_type=self.config["train_type"] #frame vedio extractor


    def init(self):
        #load models
        if(self.train_type=="frame"):
            for i in range(0,len(self.model_list)):
                self.embed_models.append(self.model_list[i](**self.model_param_list[i]))
                self.embed_models[i].load_state_dict(torch.load(
                    os.path.join(path, "model-0.pkl")))

        if(self.train_type=="extractor"):
            for i in range(0,len(self.model_list)):
                models=[]
                models.append(torchvision.models.inception_v3(pretrained=True))
                models.append(self.model_list[i](**self.model_param_list[i]))
                models[1].load_state_dict(torch.load(
                    os.path.join(path, "model-0.pkl")))
                self.embed_models.append(models)

        if(self.train_type=="vedio"):
            for i in range(0,len(self.model_list)):
                models=[]
                models.append(self.model_list[i][0](**self.model_param_list[i][1]))
                models[0].load_state_dict(torch.load(
                    os.path.join(path, "model-0.pkl")))
                models.append(self.model_list[i][1](**self.model_param_list[i][1]))
                models[1].load_state_dict(torch.load(
                    os.path.join(path, "model-1.pkl")))
                self.embed_models.append(models)


    def forward(self,x):
        pred=None
        if(self.train_type=="frame"):
            for i in range(0,len(self.embed_models)):
                if(i==0):
                    pred=self.embed_models[i](x).detach()
                else:
                    pred+=self.embed_models[i](x).detach()
            return pred
        if(self.train_type=="vedio"):
            for i in range(0,len(self.embed_models)):
                model1=self.embed_models[i][0]
                model2=self.embed_models[i][1]
                if(i==0):
                    frames=x.size(1)
                    images=x.view(-1,x.size(2),x.size(3),x.size(4))
                    features=model1(images)
                    features=features.view(-1,frames,features.size(1))
                    pred=model2(features).detach()
                else:
                    frames=x.size(1)
                    images=x.view(-1,x.size(2),x.size(3),x.size(4))
                    features=model1(images)
                    features=features.view(-1,frames,features.size(1))
                    pred+=modle2(features).detach()
            return pred

class feature_extractor_solver(solver):
    def __init__(self):
        super(feature_extractor_solver,self).__init__()

    def config(self):
        self.model_path=self.config["model_path"]
        self.dataloader=self.config["dataset_function"](**self.config["dataset_function_params"])
        self.save_path=self.config["save_path"]
        self.pca_save_path=self.config["pca_save_path"]

    def main_loop(self):
        if(self.model_path is not None):
            self.restore_params_with_path(path)
        image_ids=None
        features=None
        labels=None
        for step,data in enumerate(self.dataloader):
            image_id,x,y=data
            x=x.cuda()
            y=y.cuda()
            feature=self.models[0](x)
            if(features is None):
                image_ids=np.array(image_id)
                features=feature.detach().numpy().cpu()
                y=y.detach().numpy().cpu()
            else:
                image_ids=np.concatenate((image_ids,image_id),axis=0)
                features=np.concatenate((features,feature),axis=0)
                labels=np.concatenate((labels,y),axis=0)
        if(self.save_path is not None):
            f=h5py.File(self.save_path，"w")
            dt=h5py.special_dtype(vlen=unicode)
            f.create_dataset("feature",data=features)
            f.create_dataset("label",data=labels)
            ds=f.create_dataset("image_id",image_ids.shape,dtype=dt)
            ds[:]=image_ids
            f.close()


        #extract pca
        if(self.pca_save_path is not None):
            mean,vals,vects=utils.pca_three_value(features)
            vects=vects[:,:1024]
            np.save(os.path.join(self.pca_save_path,"mean.npy"),mean)
            np.save(os.path.join(self.pca_save_path,"eigenvals.npy"),vals)
            np.save(os.path.join(self.pca_svae_path,"eigenvecs.npy"),vects)



class detection_solver(common_solver):
    def __init__(self):
        super(detection_solver,self).__init__()

    def config(self):
        super(detection_solver,self).config()

    def train(self):
        pass
