import solver
import torch
import torch.nn as nn

#call the train_model
#the param_dict
# ["epochs"]
# ["train_loader"]
# ["val_loader"]
# ["test_loader"]


class simple_classify_solver(solver.solver):
    def __init__(self, models, model_name, save_path='checkpoints'):
        super(simple_classify_solver, self).__init__(
            models, model_name, save_path)
        self.loss_function=nn.CrossEntropyLoss()

    def test_model(self, param_dict):
        dataloader=param_dict["loader"]
        counter=0
        total_loss=0
        total_correct=0
        for step,(x,y) in enumerate(dataloader):
            counter=counter+1
            input_dict={}
            input_dict["x"]=x.cuda()
            input_dict["y"]=y.cuda()
            loss,correct=self.test_one_batch(input_dict)
            total_loss+=loss
            total_correct+=correct
        result={}
        result["test_loss"]=total_loss/counter
        result["test_acc"]=total_correct/dataloader.dataset.__len__()
        return result

    def train_one_batch(self, input_dict):
        optimizer=self.optimizers[0]
        model=self.models[0]
        x=input_dict["x"]
        y=input_dict["y"]
        out=model(x)
        pred=torch.max(out)[1]
        acc=torch.mean((pred==y).float())
        loss=self.loss_function(out,y)
        loss.backward()
        optimizer.step()
        self.zero_grad_for_all()
        total_loss={}
        total_loss["train_loss"]=loss.detach().cpu().item()
        total_loss["train_acc"]=acc.detach().cpu().item()
        return total_loss


    def test_one_batch(self, input_dict):
        model=self.models[0]
        x=input_dict["x"]
        y=input_dict["y"]
        out=model(x)
        loss=self.loss_function(out,y)
        correct=torch.sum((pred==y).float())
        return loss.detach().cpu().item(),correct.detach().cpu().item()


    def train_model(self, param_dict):
        best_acc=0.0
        iteration_count=0
        dataloader=param_dict["train_loader"]
        val_loader=param_dict["val_loader"]
        test_loader=param_dict["test_loader"]
        epochs=param_dict["epochs"]
        for i in range(0,epochs):
            for step,(x,y) in enumerate(dataloader):
                input_dict={}
                input_dict["x"]=x.cuda()
                input_dict["y"]=y.cuda()
                loss=self.train_one_batch(input_dict)
                iteration_count+=1
                if(iteration_count%1==0):
                    self.write_log(loss,iteration_count)
                    self.output_loss(loss,i,iteration_count)
            val_param_dict={}
            val_param_dict["loader"]=val_loader
            val_result=self.test_model(val_param_dict)
            self.write_log(val_result,i)
            self.output_loss(val_result,i,0)
            if(val_result["test_acc"]>best_acc):
                best_acc=val_result["test_acc"]
                self.save_params(0)
            self.update_optimizers(i)
        test_param_dict={}
        test_param_dict["loader"]=test_loader
        self.restore_params(self.time_string,0)
        test_result=self.test_model(test_param_dict)
        print("the final result is:")
        self.output_loss(val_result,i,0)
