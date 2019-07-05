import os

import cv2
import numpy as np
from skimage.measure import compare_psnr, compare_ssim


#take the tencorp data preprocess by the transform tencorp and return the pair data
#now it just support (x,y) pairs
def tencrop_process(data):
    x=data[0]
    if(len(x.size())==5):
        x=data[0]
        y=data[1]
        crop_size=x.size(1)
        list_y=[]
        size_y=list(y.size())
        size_y[0]=size_y[0]*crop_size
        y=torch.unsqueeze(y,1)
        for i in range(0,crop_size):
            list_y.append(y)
        y=torch.cat(list_y,1)
        y=y.view(size_y)
        x=x.view(-1,x.size(2),x.size(3),x.size(4))
        data=(x,y)
    return data

def calculate_acc(pred,y):
    pred_label=torch.max(pred,1)[1]
    acc=torch.sum((pred_label==y).float())/y.size(0)
    return acc.detach().cpu().item()

def pca_three_value(data):
    mean=np.mean(data,axis=0)
    data_mean=data-mean
    cov=np.cov(data_mean,rowvar=0)
    eig_vals,eig_vects=np.linalg.eig(np.mat(cov))
    eig_indice=np.argsort(-eig_vals)
    eig_vects=eig_vects[:,eig_indice]
    eig_vals=eig_vals[eig_indice]
    return mean,eig_vals,eig_vects

#the images from 0 to 255
def write_images(images,index,file_path="default_image_path"):
    if(not os.path.exists(file_path)):
        os.mkdir(file_path)
    index_path=os.path.join(file_path,int(index))
    if(not os.path.exists(index_path)):
        os.mkdir(index_path)

    if(isinstance(images,dict)):
        for k in images:
            images[k]=cv2.cvtColor(images[k],cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(index_path,str(k)+".jpg"),images[k])

    if(isinstance(images,list)):
        for k in range(0,len(images)):
            images[k]=cv2.cvtColor(images[k],cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(index_path,str(k)+".jpg"),images[k])

    print("unsupport image save data type")

def PSNR(output,target):
    return compare_psnr(target,output,data_range=1.0)

def SSIM(output,target):
    gray_out=cv2.cvtColor(output,cv2.COLOR_RGB2GRAY)
    gray_target=cv2.cvtColor(target,cv2.COLOR_RGB2GRAY)
    score,diff=compare_ssim(gray_out,gray_target,full=True,data_range=1.0)
    return score

def padding_images(images,padding_exp=16):
    height=(images.size(2)//padding_exp+1)*padding_exp
    width=(images.size(3)//padding_exp+1)*padding_exp
    images=F.pad(images,(0,width,0,height))
    return images

def cut_images(images,padding_exp=32):
    hegith=images.size(2)-images.size(2)%padding_exp
    width=images.size(3)-images.size(3)%padding_exp
    return images[:,:,:height,:width]
