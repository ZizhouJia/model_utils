import solver
import torch
import torch.nn as nn

class common_solver(solver.solver):
    def __init__(self, models,optimizers,kernel_processer,model_name,save_path='checkpoints'):
        super(common_solver, self).__init__(
            models,optimizers,kernel_processer,model_name,save_path)

    def test_model(self, param_dict,mode="test"):
        loader_choice={"test":"test_loader","val":"val_loader"}
        self.eval_mode()
        dataloader=param_dict[loader_choice[mode]]
        counter=0.0
        evaluate_value=0.0
        evaluate_dict=None
        for step,data in enumerate(dataloader):
            #cuda mode
            if self.is_cuda:
                for i in range(0,len(data)):
                    data[i]=data[i].cuda()
            data_counter,key_value,output_dict=self.kernel_processer.test(step,data)
            counter+=data_counter
            evaluate_value+=(key_value*data_counter)
            if(evaluate_dict is None):
                evaluate_dict={}
                for key in output_dict.keys():
                    evaluate_dict[key]=output_dict[key]*counter
            else:
                for key in output_dict.keys():
                    evaluate_dict[key]+=(output_dict[key]*counter)
        for key in evaluate_dict.keys():
            evaluate_dict[key]=evaluate_dict[key]/counter
        evaluate_value=evaluate_value/counter
        return evaluate_value,evaluate_dict


    def train_model(self,epoch,param_dict):
        self.train_mode()
        dataloader=param_dict["train_loader"]
        dataset_numbers=dataloader.dataset.__len__()
        for step,data in enumerate(dataloader):
            if self.is_cuda:
                for i in range(0,len(data)):
                    data[i]=data[i].cuda()
            evaluate_dict=self.kernel_processer.train(step,data)
            self.write_log(evaluate_dict,epoch*dataset_numbers+step)
            self.output_loss(evaluate_dict,epoch,step)
            self.kernel_processer.update_optimizers(epoch,step,dataset_numbers)

    #param_dict ["train_loader","val_loader","test_loader"]
    def main(self,param_dict):
        best_value=1e10
        iteration_count=0
        epochs=param_dict["epochs"]
        total_data_numbers=param_dict["train_loader"].__len__()
        for i in range(0,epochs):
            self.train_model(i,param_dict)
            evaluate_value,evaluate_dict=self.test_model(param_dict,"val")
            self.output_loss(evaluate_dict,epoch,0)
            self.write_log(evaluate_dict,epoch,step)
            if(evaluate_value<best_value):
                best_value=evaluate_value
                self.save_params("best")
        self.restore_params(self.time_string,"best")
        tev,ted=self.test_model(param_dict,"test")
        return tev
