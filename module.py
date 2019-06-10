import torch
import torch.nn as nn
import torch.nn.functional as F


class MultBoxLoss(nn.Module):
    def __init__(self,num_class,overlap_thresh,prior_for_matching,
    bkg_label,neg_mining,neg_pos,neg_overlap,encode_target,variance):
        super(MultiBoxLoss,self).__init__()
        self.num_classes=num_class
        self.threshold=overlap_thresh
        self.background_label=bkg_label
        self.encode_target=encode_target
        self.use_prior_for_matching=prior_for_matching
        self.do_neg_mining=neg_mining
        self.negpos_ratio=neg_pos
        self.neg_overlap=neg_overlap
        self.variance=variance
    def forward(self,preddictions,targets):
        local_data,conf_data,priors=preddictions
        num=local_data.size(0)
        priors=priors[:local_data.size(1),:]
        num_priors=priors.size(0)
        num_classes=self.num_classes

        loc_t=torch.Tensor(num,num_priors,4)
        conf_t=torch.LongTensor(num,num_priors)
        for idx in range(num):
            truths=targets[idx][:,:-1].data
            labels=targets[idx][:,-1].data
            defaults=priors.data
            match(self.threshold,truths,defaults,self.variance,labels,loc_t,conf_t,idx)

class distilling_classify_loss(nn.Module):
    def __init__(self,lamda=0.5):
        super(self,distilling_classify_loss).__init__()
        self.lamda=lamda

    def forward(self,pred,y,soft_y):
        pred=F.sigmoid(pred)
        loss=-(self.lamda*y+(1-self.lamda)*y_soft)*torch.log(pred)-(self.lamda*(1-y)+(1-self.lamda)*(1-soft_y))*torch.log(1-pred)
        loss=torch.mean(loss)
        return loss
