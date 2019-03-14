import numpy as np
import random
import torch
import os
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#the dict dict["file name"]=label
class dict_dataset(torch.utils.data.Dataset):
    def __init__(self,dataset_dict,path,mode,transform,percent=[0.7,0.1,0.2],shuffle=True):
        self.dataset_dict=dataset_dict
        self.path=path
        self.mode=mode
        self.dataset_dict_list=self.dataset_dict.keys()
        self.transform=transform[self.mode]
        self.train_data_numbers=int(len(self.dataset_dict_list)*percent[0])
        self.val_data_numbers=int(len(self.dataset_dict_list)*percent[1])
        self.test_data_numbers=len(self.dataset_dict_list)-self.train_data_numbers-self.val_data_numbers

    def _read_image(self,key_name):
        image_name=os.path.join(self.path,key_name)
        fp=open(image_name)
        image=Image.open(fp).convert('RGB')
        return image

    def _mapping_index(self,index):
        actual_index=index
        if(self.mode=="train"):
            return actual_index
        if(self.mode=="val"):
            return actual_index+self.train_data_numbers
        if(self.mode=="test"):
            return actual_index+self.train_data_numbers+self.val_data_numbers
        return actual_index

    def __getitem__(self,index):
        actual_index=self._mapping_index(index)
        key_name=self.dataset_dict_list[actual_index]
        label=self.dataset_dict[key_name]
        image=self._read_image(key_name)
        image=self.transform(image)
        return image,label

    def __len__(self,index):
        if(mode=="train"):
            return self.train_data_numbers
        if(mode=="val"):
            return self.val_data_numbers
        if(mode=="test"):
            return self.test_data_numbers
