import utils
import trainloaded
import embtraining
import torch
import pandas as pd

x, y, mn, std, xt = utils.loaddata('NAS', 0, dir="./data/", raw=True)
ypreles = pd.read_csv('./data/soro_p.csv')
yp = ypreles['GPPp']


print(x,y,mn,std,xt)

xn = xt.drop(['date', 'GPP', 'ET'], axis=1)
yy = y.drop(['ET'], axis=1)

hpar = {'epochs': 100,
        'batchsize': 2,
        'learningrate': 0.001}

model_design = {'layer_sizes': [[16, 16], [128]]}
#loss = embtraining.train(hpar, model_design, X=x.to_numpy(), Y=yy.to_numpy(), Xn=xn.to_numpy(), mean=mn, std=std, pre=yp)
#print('tv', loss['train_loss'], 'vv', loss['val_loss'])
#pd.DataFrame(loss).to_csv("./lossEMBbNAS.csv")

#pretrained = "./EMBmodelNAS.pth"


loss = embtraining.train(hpar, model_design, X=x.to_numpy(), Y=yy.to_numpy(), Xn=xn.to_numpy(), mean=mn, std=std, pre=yp)#, pt=pretrained)
pd.DataFrame(loss).to_csv("./lossEMBbNAS.csv")


#embtraining.train()


