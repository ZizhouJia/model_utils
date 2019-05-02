#the super version of dataloader and dataset
import numpy as np
import torch.multiprocessing as multiprocessing

class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class collect_fn(object):
    def __init__(self):
        pass

    def __call__(self,data):
        item_nums=len(data[0])
        return_list=[]
        for i in range(0,item_nums):
            return_list.append([])
        for item in range(0,item_nums):
            for i in range(0,len(data)):
                return_list[item].append(data[i][item])
        format_list=[]
        for item in range(0,item_nums):
            if(isinstance(data[0][item],str)):
                format_list.append(tuple(return_list[item]))
                continue
            if(isinstance(data[0][item],int)):
                format_list.append(torch.Tensor(return_list[item]))
                continue
            if(isinstance(data[0][item],float)):
                format_list.append(torch.Tensor(return_list[item]))
                continue
            if(isinstance(data[0][item],torch.Tensor)):
                format_list.append(torch.cat(return_list[item],dim=0))
                continue
            if(isinstance(data[0][item],np.Array)):
                format_list.append(np.concatenate(return_list[item],axis=0))
                continue
            raise RuntimeError("unsupport data type")




#accelerate the dataloader with the buffer for show read dataset
#note the BufferDataLoader just support many num_workers if the num_workers set to be 0,it mean the 1 worker
class BufferDataLoader(object):
    def __init__(self,dataset,batch_size=1,shuffle=False,num_workers=0,drop_last=False,collect_fn=None,buffer_size=100):
        self.dataset=dataset
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.num_workers=num_workers
        self.drop_last=drop_last
        self.buffer_size=buffer_size
        if(self.num_workers<=0):
            self.num_workers=1
        if(collect_fn is None):
            self.collect_fn=collect_fn()
        else:
            self.collect_fn=collect_fn


    def __iter__(self):
        return _BufferDataLoaderIter(self)

    def __len__(self):
        return len(self.dataset)


def _data_worker(dataset,index_queue,data_buffer):
    while(True):
        index=index_queue.get()
        data_buffer.put(dataset.__getitem__(index))


class _BufferDataLoaderIter(object):
    def __init__(self,loader):
        self.dataset=loader.dataset
        self.batch_size=loader.batch_size
        self.shuffle=loader.shuffle
        self.num_workers=loader.num_workers
        self.buffer_size=loader.buffer_size
        self.drop_last=loader.drop_last
        self.collect_fn=loader.collect_fn
        self.loop_mode=False
        #init worker
        self.data_buffer=multiprocessing.Queue(max_size=self.buffer_size)
        self.index_queue=multiprocessing.Queue(max_size=self.buffer_size)
        self.index_queue.cancel_join_thread()
        self.indexs=self._init_indexs()
        self.buffer_index=0
        self.current_index=0
        self.workers=[]
        self.buffer_end=False
        for i in range(0,num_workers):
            w=multiprocessing.Process(target=_data_worker,args=(self.dataset,self.index_queue,self.data_buffer))
            w.start()
            self.workers.append(w)

    def _fill_index_queue(self):
        if(self.buffer_end):
            return
        while(not self.index_queue.full()):
            if(self.buffer_index==len(self.indexs)):
                self.indexs=self._init_indexs()
                self.buffer_index=0
                if(self.loop_mode):
                    continue
                else:
                    self.buffer_end=True
                    break
            self.index_queue.put(self.indexs[self.buffer_index])
            self.buffer_index+=1


    def _init_indexs(self):
        indexs=np.arange(len(self.dataset))
        if(self.shuffle):
            np.random.shuffle(indexs)
        return indexs

    def set_loop_mode(self):
        self.loop_mode=True

    def __next__(self):
        if(self.loop_mode):
            self.current_index=(self.current_index+self.batch_size)%len(self.dataset)
            data_list=[]
            for i in range(0,self.batch_size):
                data_list.append(self.data_buffer.get())
            self._fill_index_queue()
            return self.collect_fn(data_list)

        if(self.drop_last and len(self.dataset)-self.current_index<self.batch_size):
            raise StopIteration

        if(self.current_index==len(self.dataset)):
            self.current_index=0
            self.buffer_end=False
            self._fill_index_queue()
            raise StopIteration

        ret_num=0
        if(len(self.dataset)-self.current_index<self.batch_size):
            ret_num=len(self.dataset-self.current_index)
            self.current_index=len(self.dataset)
        else:
            ret_num=self.batch_size
            self.current_index+=self.batch_size

        data_list=[]
        for i in range(0,ret_num):
            data_list.append(self.data_buffer.get())
        self._fill_index_queue()
        return self.collect_fn(data_list)



    def __iter__(self):
        return self


    def __len__(self):
        return len(self.dataset)

    next=__next__

    def __del__(self):
        self.data_buffer.cancel_join_thread()
        self.index_queue.cancel_join_thread()
        self.data_buffer.close()
        self.index_queue.close()
        for w in self.workers:
            w.close()
