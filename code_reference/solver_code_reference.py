import numpy as np
import torch
import torch.nn as nn

from . import draw, module, solver


class common_classify_solver(solver.common_solver):
    def __init__(self):
        super(common_classify_solver, self).__init__()
        self.loss_function = nn.CrossEntropyLoss()
        self.pred = []
        self.label = []
        self.image_name = []
        self.loss_value = []
        self.element_number = []
        self.best_acc = 0.0

    def empty_list(self):
        self.pred = []
        self.label = []
        self.image_name = []
        self.loss_value = []
        self.element_number = []

    def before_test(self):
        if(self.if_train):
            self.restore_params(self.task_name, self.timestemp, "best")

    def before_train(self):
        if(self.request.epoch in self.learning_rate_decay_epochs):
            for optimizer in self.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*0.1

    def train(self):
        model = self.models[0]
        optimizer = self.optimizers[0]
        x, y, _ = self.request.data
        x = x.cuda()
        y = y.cuda()
        pred = model(x)
        loss = self.loss_function(pred, y)
        loss.backward()
        optimizer.step()
        self.zero_grad_for_all()
        if(self.request.iteration % 20 == 0):
            show_dict = {}
            pred_label = torch.max(pred, 1)[1]
            acc = torch.sum((pred_label == y).float())/x.size(0)
            show_dict["train_acc"] = acc.detach().cpu().item()
            show_dict["train_loss"] = loss.detach().cpu().item()
            self.write_log(show_dict, self.request.iteration)
            self.print_log(show_dict, self.request.epoch, self.request.step)

    def validate(self):
        model = self.models[0]
        x, y, image_name = self.request.data
        x = x.cuda()
        y = y.cuda()
        pred = model(x)
        loss = self.loss_function(pred, y)
        pred = pred.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        for i in range(0, pred.shape[0]):
            self.pred.append(pred[i, :])
            self.label.append(y[i])
            self.image_name.append(image_name[i])
        self.loss_value.append(loss.detach().cpu().item())
        self.element_number.append(x.size(0))

    def test(self):
        self.validate()

    def after_validate(self):
        preds = np.array(self.pred)
        labels = np.array(self.label)
        pred_label = np.argmax(preds, axis=1)
        acc = np.sum((pred_label == labels).astype(np.float))/preds.shape[0]
        total_loss = 0
        total_emement = 0
        for i in range(0, len(self.loss_value)):
            total_loss += (self.loss_value[i]*self.element_number[i])
            total_emement += self.element_number[i]
        loss = total_loss/total_emement
        write_dict = {}
        write_dict["validate_loss"] = loss
        write_dict["validate_acc"] = acc
        self.write_log(write_dict, self.request.epoch)
        self.print_log(write_dict, self.request.epoch, 0)
        self.empty_list()
        if(write_dict["validate_acc"] > self.best_acc):
            self.best_acc = write_dict["validate_acc"]
            self.save_params("best")

    def after_test(self):
        preds = np.array(self.pred)
        labels = np.array(self.label)
        pred_label = np.argmax(preds, axis=1)
        acc = np.sum((pred_label == labels).astype(np.float))/preds.shape[0]
        total_loss = 0
        total_emement = 0
        for i in range(0, len(self.loss_value)):
            total_loss += self.loss_value[i]*self.element_number[i]
            total_emement += self.element_number[i]
        loss = total_loss/total_emement
        confusion_matrix = np.zeros((preds.shape[1], preds.shape[1]))
        for i in range(0, len(pred_label)):
            confusion_matrix[labels[i], pred_label[i]] += 1

        write_dict = {}
        write_dict["test_loss"] = loss
        write_dict["test_acc"] = acc
        self.write_log(write_dict, 0)
        self.print_log(write_dict, self.request.epoch, 0)
        draw.plotCM(confusion_matrix, "confusion_matrix.jpg")
        f = open("pred.txt", "w")
        for i in range(0, len(self.image_name)):
            f.write(self.image_name[i]+" "+str(pred_label[i])+" \n")
        f.close()
        self.empty_list()


class t_sne_solver(solver.common_solver):
    def __init__(self):
        super(t_sne_solver, self).__init__()
        self.features = []
        self.labels = []

    def main_loop(self):
        self.load_config()
        model = self.models[0]
        for step, data in enumerate(self.test_loader):
            x, y, _ = data
            x = x.cuda()
            y = y.cuda()
            print(step)
            feature = model(x)
            feature = feature.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            for i in range(0, feature.shape[0]):
                self.features.append(feature[i])
                self.labels.append(y[i])
        # begin t-SNE
        draw.plotTSNE(np.array(self.features),
                      np.array(self.labels), "t_SNE.jpg")


class vedio_classify_solver(solver.common_solver):
    def __init__(self):
        super(vedio_classify_solver, self).__init__()
        self.loss_function = nn.CrossEntropyLoss()
        self.pred = []
        self.label = []
        self.image_name = []
        self.loss_value = []
        self.element_number = []
        self.best_acc = 0.0

    def empty_list(self):
        self.pred = []
        self.label = []
        self.image_name = []
        self.loss_value = []
        self.element_number = []

    def load_config(self):
        super(vedio_classify_solver, self).load_config()
        self.learning_rate_decay_iteration = self.config["learning_rate_decay_iteration"]
        self.grad_plenty = self.config["grad_plenty"]
        # default is False
        self.distilling_mode = self.config["distilling_mode"]
        if(self.distilling_mode):
            self.distilling_loss = module.distilling_classify_loss(0.5)

    def init(self):
        if(len(self.models) == 1):
            self.train_type = "frame"
        else:
            self.train_type = "vedio"

    def forward(self, x):
        if(self.train_type == "frame"):
            pred = self.models[0](x)
            return pred
        if(self.train_type == "vedio"):
            frames = x.size(1)
            x = x.view(-1, x.size(2), x.size(3), x.size(4))
            features = self.models[0](x)
            features = features.view(-1, frames, features.size(1))
            x = features
            pred = self.models[1](x)
            return pred

    def request_data(self):
        if(self.distilling_mode):
            _, x, y, soft_y = self.request.data
            return x, y, soft_y
        else:
            _, x, y = self.request.data
            return x, y, None

    def train(self):
        x, y, soft_y = self.request_data()
        x = x.cuda()
        y = y.cuda()
        pred = self.forward(x)
        loss = None
        if(soft_y is None):
            loss = self.loss_function(pred, y)
        else:
            loss = self.distilling_loss(pred, y, soft_y)

        loss.backward()
        if(self.grad_plenty != 0):
            nn.utils.clip_grad_norm_(
                self.models[0].parameters(), self.grad_plenty, norm_type=2)
        self.optimize_all()
        self.zero_grad_for_all()
        if(self.request.iteration % 10 == 0):
            show_dict = {}
            pred_label = torch.max(pred, 1)[1]
            # print(pred_label)
            # print(y)
            acc = torch.sum((pred_label == y).float())/x.size(0)
            show_dict["train_acc"] = acc.detach().cpu().item()
            show_dict["train_loss"] = loss.detach().cpu().item()
            self.write_log(show_dict, self.request.iteration)
            self.print_log(show_dict, self.request.epoch, self.request.step)

        if(self.learning_rate_decay_iteration != 0):
            if((self.request.iteration+1) % self.learning_rate_decay_iteration == 0):
                for optimizer in self.optimizers:
                    for param_group in optimizer:
                        param_group['lr'] = param_group['lr']*0.95

    def after_train(self):
        if(self.request.epoch in self.learning_rate_decay_epochs):
            for optimizer in self.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*0.1

    def validate(self):
        x, y, soft_y = self.request_data()
        x = x.cuda()
        y = y.cuda()
        pred = self.forward(x)
        loss = None
        if(soft_y is None):
            loss = self.loss_function(pred, y)
        else:
            loss = self.distilling_loss(pred, y, soft_y)
        pred = pred.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        for i in range(0, pred.shape[0]):
            self.pred.append(pred[i, :])
            self.label.append(y[i])
            # self.image_name.append(image_name[i])
        self.loss_value.append(loss.detach().cpu().item())
        self.element_number.append(x.size(0))

    def test(self):
        self.validate()

    def after_validate(self):
        preds = np.array(self.pred)
        labels = np.array(self.label)
        pred_label = np.argmax(preds, axis=1)
        acc = np.sum((pred_label == labels).astype(np.float))/preds.shape[0]
        total_loss = 0
        total_emement = 0
        for i in range(0, len(self.loss_value)):
            total_loss += (self.loss_value[i]*self.element_number[i])
            total_emement += self.element_number[i]
        loss = total_loss/total_emement
        write_dict = {}
        write_dict["validate_loss"] = loss
        write_dict["validate_acc"] = acc

        confusion_matrix = np.zeros((preds.shape[1], preds.shape[1]))
        for i in range(0, len(pred_label)):
            confusion_matrix[labels[i], pred_label[i]] += 1
        print(confusion_matrix)
        tfpn_matrix = np.zeros((2, 2))
        tfpn_matrix[0, 0] = confusion_matrix[0, 0]+confusion_matrix[0,
                                                                    1]+confusion_matrix[1, 0]+confusion_matrix[1, 1]
        tfpn_matrix[1, 1] = confusion_matrix[2, 2]
        tfpn_matrix[1, 0] = confusion_matrix[2, 0]+confusion_matrix[2, 1]
        tfpn_matrix[0, 1] = confusion_matrix[0, 2]+confusion_matrix[1, 2]
        print(tfpn_matrix)
        write_dict["validate_pression"] = tfpn_matrix[1, 1] / \
            (tfpn_matrix[1, 1]+tfpn_matrix[0, 1])
        write_dict["validate_recall"] = tfpn_matrix[1, 1] / \
            (tfpn_matrix[1, 1]+tfpn_matrix[1, 0])
        write_dict["validate_f_score"] = write_dict["validate_pression"] * \
            write_dict["validate_recall"] / \
            (write_dict["validate_pression"]+write_dict["validate_recall"])
        self.write_log(write_dict, self.request.epoch)
        self.print_log(write_dict, self.request.epoch, 0)
        self.empty_list()
        if(write_dict["validate_f_score"] > self.best_acc):
            self.best_acc = write_dict["validate_f_score"]
            self.save_params("best")

    def before_test(self):
        if(self.if_train):
            self.restore_params(self.task_name, self.timestemp, "best")

    def after_test(self):
        preds = np.array(self.pred)
        labels = np.array(self.label)
        pred_label = np.argmax(preds, axis=1)
        acc = np.sum((pred_label == labels).astype(np.float))/preds.shape[0]
        total_loss = 0
        total_emement = 0
        for i in range(0, len(self.loss_value)):
            total_loss += self.loss_value[i]*self.element_number[i]
            total_emement += self.element_number[i]
        loss = total_loss/total_emement
        confusion_matrix = np.zeros((preds.shape[1], preds.shape[1]))
        for i in range(0, len(pred_label)):
            confusion_matrix[labels[i], pred_label[i]] += 1
        print(confusion_matrix)

        write_dict = {}

        tfpn_matrix = np.zeros((2, 2))
        tfpn_matrix[0, 0] = confusion_matrix[0, 0]+confusion_matrix[0,
                                                                    1]+confusion_matrix[1, 0]+confusion_matrix[1, 1]
        tfpn_matrix[1, 1] = confusion_matrix[2, 2]
        tfpn_matrix[1, 0] = confusion_matrix[2, 0]+confusion_matrix[2, 1]
        tfpn_matrix[0, 1] = confusion_matrix[0, 2]+confusion_matrix[1, 2]
        print(tfpn_matrix)
        write_dict["test_pression"] = tfpn_matrix[1, 1] / \
            (tfpn_matrix[1, 1]+tfpn_matrix[0, 1])
        write_dict["test_recall"] = tfpn_matrix[1, 1] / \
            (tfpn_matrix[1, 1]+tfpn_matrix[1, 0])
        write_dict["test_f_score"] = write_dict["test_pression"] * \
            write_dict["test_recall"] / \
            (write_dict["test_pression"]+write_dict["test_recall"])

        write_dict = {}
        write_dict["test_loss"] = loss
        write_dict["test_acc"] = acc
        self.write_log(write_dict, 0)
        self.print_log(write_dict, self.request.epoch, 0)
        draw.plotCM(confusion_matrix, self.task_name+"_confusion_matrix.jpg")
        draw.plotCM(tfpn_matrix, self.task_name+"_tfpn_matrix.jpg")
        self.empty_list()


class detection_solver(solver.common_solver):
    def __init__(self):
        super(detection_solver, self).__init__()

    def load_config(self):
        super(detection_solver, self).load_config()

    def train(self):
        pass
