import solver
import torch
import torch.nn as nn


class simple_regression_solver(sovler.sovler):
    def __init__(self, models, model_name, save_path='checkpoints'):
        super(simple_classify_solver, self).__init__(
            models, model_name, save_path)
        self.mse=nn.MSELoss()


    def test_model(self, param_dict):
        dataloader=param_dict["loader"]
        counter=0
        total_loss=0
        for step,(x,y) in enumerate(dataloader):
            counter=counter+1
            input_dict={}
            input_dict["x"]=x
            input_dict["y"]=y
            loss=self.test_one_batch(input_dict)
            total_loss+=loss
        print("the test loss is: %.4f"%(total_loss/counter))


    def train_one_batch(self, input_dict):
        optimizer=self.optimizers[0]
        model=self.model[0]
        x=input_dict["x"]
        y=input_dict["y"]
        out=model(x)
        loss=self.mse(out,y)
        loss.backward()
        optimizer.step()
        self.zero_grad_for_all()
        total_loss={}
        total_loss["loss"]=loss.detach().cpu().item()
        return total_loss


    def test_one_batch(self, input_dict):
        x=input_dict["x"]
        y=input_dict["y"]
        out=model(x)
        loss=self.mse(out,y)
        return loss.detach().cpu().item()


    def train_model(self, param_dict):
        iteration_count=0
        dataloader=param_dict["loader"]
        for i in range(0,epochs):
            for step,(x,y) in enumerate(dataloader):
                input_dict={}
                input_dict["x"]=x
                input_dict["y"]=y
                loss=self.train_one_batch(input_dict)
                iteration_count+=1
                if(iteration_count%100==0):
                    self.write_log(loss,iteration_count)
                    self.output_loss(loss,i,iteration_count)
