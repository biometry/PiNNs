import utils
import embtraining
import torch


x, y, xt = utils.loaddata('NAS', 0, dir="./data/", raw=True)
xn = xt.drop(['date', 'GPP', 'ET'], axis=1)

hpar = {'epochs': 100,
        'batchsize': 8,
        'learningrate': 0.001,
}

model_design = {'layer_sizes': [[16], [8]]}
embtraining.train(hpar, model_design, X=x.to_numpy(), Y=y.to_numpy(), Xn=xn.to_numpy())

#pretrained = torch.load("./model.pth")


#embtraining.train(hpar, model_design, X=x.to_numpy(), Y=y.to_numpy(), Xn=xn.to_numpy(), pt = pretrained)



#embtraining.train()

