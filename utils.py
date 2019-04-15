
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
