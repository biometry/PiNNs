# !/usr/bin/env python
# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import preles

# naive feed forward MLP
class NMLP(nn.Module):

    def __init__(self, input_dim, output_dim, layersizes):
        super(NMLP, self).__init__()

        self.layers = nn.Sequential()
        self.nlayers = len(layersizes)
        print('Initializing model')
        for i in range(0, self.nlayers+1):
            if i == 0:
                self.layers.add_module(f'input{i}', nn.Linear(input_dim, layersizes[i]))
                self.layers.add_module(f'activation{i}', nn.ReLU())
                print('adding input l', input_dim, layersizes[i])
            elif i == (self.nlayers):
                self.layers.add_module(f'output{i}', nn.Linear(layersizes[i-1], output_dim))
                print('adding output l', layersizes[i-1], output_dim)
            elif i and i < self.nlayers:
                self.layers.add_module(f'fc{i}', nn.Linear(layersizes[i-1], layersizes[i]))
                self.layers.add_module(f'activation{i}', nn.ReLU())
                print('adding hidden l', layersizes[i-1], layersizes[i])
                

    def forward(self, x):
        out = self.layers(x)

        return out




class EMB(nn.Module):

    def __init__(self, input_dim, output_dim, layersizes, pin, pout):
        super(EMB, self).__init__()

        self.parnet = nn.Sequential()
        self.resnet = nn.Sequential()
        self.pnlayers = len(layersizes[0])
        self.preleslayer = [pin, pout]
        self.rnlayers = len(layersizes[1])
        print('Initializing Model')
        print(self.pnlayers, self.rnlayers)
        # Add Parameter Layers

        for i in range(0, self.pnlayers+1):
            if i == 0:
                self.parnet.add_module(f'P_input{i}', nn.Linear(input_dim, layersizes[0][i]))
                self.parnet.add_module(f'activation', nn.ELU())
                print('adding input 1', input_dim, layersizes[0][i])
            elif i == (self.pnlayers):
                print("i", i, "layers", layersizes[0][i-1])
                self.parnet.add_module(f'P_output{i}', nn.Linear(layersizes[0][i-1], pin))
                print("add parameters output", layersizes[0][i-1])
            elif i and i < self.pnlayers:
                self.parnet.add_module(f'P_fc{i}', nn.Linear(layersizes[0][i-1], layersizes[0][i]))
                self.parnet.add_module(f'activation', nn.ELU())


        # Add Residual Layers
        for i in range(0, self.rnlayers+1):
            if i == 0:
                self.resnet.add_module(f'R_input{i}', nn.Linear(pout, layersizes[1][i]))
                self.resnet.add_module(f'activation', nn.ELU())
                print('adding input 1', pout, layersizes[1][i])
            elif i == self.rnlayers:
                self.resnet.add_module(f'P_output{i}', nn.Linear(layersizes[1][i-1], output_dim))
                print("add parameters output", layersizes[1][i-1])
            elif i and i < self.rnlayers:
                self.resnet.add_module(f'P_fc{i}', nn.Linear(layersizes[1][i-1], layersizes[1][i]))
                self.resnet.add_module(f'activation', nn.ELU())

        # Initialize weights of Parameter Layer


    def forward(self, x, cin, tp=None, sw=None):
        pinit = self.parnet(x)
        #print(pinit)
        #p = sparameter_constraint(pinit.type(torch.float64))
        ppreds = physical_forward(pinit, cin, tp, sw)
        y_hat = self.resnet(ppreds.type(torch.float32))
        return y_hat, ppreds.type(torch.float32)

    

def physical_forward(parameters, input_data, tp=None, sw=None):

    # extract Parameters
    p1 = torch.mean(parameters[...,0:1], dtype=torch.float64) + 413  #mean(parameters[...,0:1], dtype=torch.float64) + 413 #torch.tensor(413, dtype=torch.float64) #torch.mean(parameters[..., 0:1], dtype=torch.float64) + 413
    p2 = torch.tensor(0.45, dtype=torch.float64) #tensor(0.45, dtype=torch.float64) #torch.mean(parameters[..., 1:2], dtype=torch.float64) #torch.tensor(0.45, dtype=torch.float64) #torch.mean(parameters[..., 1:2], dtype=torch.float64)
    p3 = torch.tensor(0.118, dtype=torch.float64) #torch.mean(parameters[..., 2:3], dtype=torch.float64) #torch.tensor(0.118, dtype=torch.float64) #torch.mean(parameters[..., 2:3], dtype=torch.float64)
    p4 = torch.tensor(3, dtype=torch.float64) #mean(parameters[...,1:2], dtype=torch.float64) + 3 #tensor(3, dtype=torch.float64) #torch.mean(parameters[..., 3:4], dtype=torch.float64)
    p5 = torch.mean(parameters[...,1:2], dtype=torch.float64) + 1 #tensor(0.748018, dtype=torch.float64) #torch.mean(parameters[..., 4:5], dtype=torch.float64)
    p6 = torch.tensor(10.93, dtype=torch.float64) #mean(parameters[...,3:4], dtype=torch.float64) + 11 #tensor(10.93, dtype=torch.float64) #torch.mean(parameters[..., 5:6], dtype=torch.float64)*12 #torch.tensor(13.233830, dtype=torch.float64) #torch.mean(parameters[..., 5:6], dtype=torch.float64)*12
    p7 = torch.tensor(-3.063, dtype=torch.float64) #mean(parameters[...,2:3], dtype=torch.float64) - 3 #tensor(-3.063, dtype=torch.float64) #torch.mean(parameters[..., 6:7], dtype=torch.float64)*10  #torch.tensor(-3.965787, dtype=torch.float64)
    p8 = torch.mean(parameters[..., 2:3], dtype=torch.float64)+17  #17.72 torch.tensor(18.766960, dtype=torch.float64) #torch.mean(parameters[..., 7:8], dtype=torch.float64)*10
    p9 = torch.mean(parameters[..., 3:4], dtype=torch.float64) #tensor(-0.1027, dtype=torch.float64) #torch.mean(parameters[..., 8:9], dtype=torch.float64) #torch.tensor(-0.130473, dtype=torch.float64) #torch.mean(parameters[..., 8:9], dtype=torch.float64)
    p10 = torch.tensor(0.03673, dtype=torch.float64) #torch.mean(parameters[..., 9:10], dtype=torch.float64) #torch.tensor(0.034459, dtype=torch.float64) #torch.mean(parameters[..., 9:10], dtype=torch.float64)
    p11 = torch.mean(parameters[..., 4:5], dtype=torch.float64) + 0.8 #tensor(0.7779, dtype=torch.float64) #torch.mean(parameters[..., 10:11], dtype=torch.float64) #torch.tensor(0.450828, dtype=torch.float64) #torch.mean(parameters[..., 10:11], dtype=torch.float64)
    p12 = torch.tensor(0.5, dtype=torch.float64) #mean(parameters[...,2:3], dtype=torch.float64) #tensor(0.5, dtype=torch.float64) #torch.mean(parameters[..., 11:12], dtype=torch.float64)
    p13 = torch.tensor(-0.364, dtype=torch.float64) #torch.mean(parameters[..., 12:13], dtype=torch.float64)
    p14 = torch.mean(parameters[...,5:6], dtype=torch.float64) #tensor(0.2715, dtype=torch.float64) #torch.mean(parameters[..., 13:14], dtype=torch.float64)*10 #torch.tensor(0.324463, dtype=torch.float64) #torch.mean(parameters[..., 13:14], dtype=torch.float64)*10
    p15 = torch.mean(parameters[...,6:7], dtype=torch.float64) +1  #tensor(0.8351, dtype=torch.float64) #torch.mean(parameters[..., 14:15], dtype=torch.float64)
    p16 = torch.mean(parameters[...,7:8], dtype=torch.float64) #tensor(0.07348, dtype=torch.float64) #torch.mean(parameters[..., 15:16], dtype=torch.float64)
    p17 = torch.tensor(0.9996, dtype=torch.float64) #mean(parameters[...,3:4], dtype=torch.float64) + 1 #tensor(0.9996, dtype=torch.float64) #torch.mean(parameters[..., 16:17], dtype=torch.float64)
    p18 = torch.mean(parameters[...,8:9], dtype=torch.float64) + 0.5 #tensor(0.4428, dtype=torch.float64) #torch.mean(parameters[..., 17:18], dtype=torch.float64)*5
    p19 = torch.mean(parameters[...,9:10], dtype=torch.float64) + 1 #tensor(1.2, dtype=torch.float64) #torch.mean(parameters[..., 18:19], dtype=torch.float64) 1.2
    p20 = torch.mean(parameters[...,10:11], dtype=torch.float64) #tensor(0.33, dtype=torch.float64) #torch.mean(parameters[..., 19:20], dtype=torch.float64)
    p21 = torch.mean(parameters[...,11:12], dtype=torch.float64)+ 5 #tensor(4.970496, dtype=torch.float64) #mean(parameters[...,4:5], dtype=torch.float64) + 5 #tensor(4.970496, dtype=torch.float64) #torch.mean(parameters[..., 20:21], dtype=torch.float64)*4
    p22 = torch.tensor(0, dtype=torch.float64) #torch.mean(parameters[..., 21:22], dtype=torch.float64)
    p23 = torch.tensor(0, dtype=torch.float64) #torch.mean(parameters[..., 22:23], dtype=torch.float64)
    p24 = torch.tensor(160, dtype=torch.float64) #torch.mean(parameters[..., 23:24], dtype=torch.float64)*180
    p25 = torch.tensor(0, dtype=torch.float64) #torch.mean(parameters[..., 24:25], dtype=torch.float64)
    p26 = torch.tensor(0, dtype=torch.float64) #torch.mean(parameters[..., 25:26], dtype=torch.float64)
    p27 = torch.tensor(0, dtype=torch.float64) #torch.mean(parameters[..., 26:27], dtype=torch.float64)*10
    #p28 = torch.mean(parameters[..., 27:28], dtype=torch.float64)
    #p29 = torch.mean(parameters[..., 28:29], dtype=torch.float64)
    #p30 = torch.mean(parameters[..., 29:30], dtype=torch.float64)

    p28 = torch.tensor(-999., dtype=torch.float64)
    p29 = torch.tensor(-999., dtype=torch.float64)
    p30 = torch.tensor(-999., dtype=torch.float64)

    Precip = torch.flatten(input_data[..., 0:1]).type(torch.float64)
    TAir = torch.flatten(input_data[..., 1:2]).type(torch.float64)
    PAR = torch.flatten(input_data[..., 2:3]).type(torch.float64)
    VPD = torch.flatten(input_data[..., 3:4]).type(torch.float64)
    fAPAR = torch.flatten(input_data[..., 4:5]).type(torch.float64)
    DOY = torch.flatten(input_data[..., 5:6]).type(torch.float64)
    CO2 = torch.flatten(input_data[..., 6:7]).type(torch.float64)
    leng = len(PAR)

    GPPmeas = torch.tensor([-999.]*leng, requires_grad=False)
    ETmeas = torch.tensor([-999.]*leng, requires_grad=False)
    SWmeas = torch.tensor([-999.]*leng, requires_grad=False)
    transp = torch.tensor([-999.]*leng, requires_grad=False)
    evap = torch.tensor([-999.]*leng, requires_grad=False)
    fWE = torch.tensor([-999.]*leng, requires_grad=False)

    logflag = 0
    control = 0
    NofDays = leng

    # Outputs
    GPP = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=True)
    ET = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=True)
    SW = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=False)
    SOG = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=False)
    fS = torch.tensor([0]*leng, dtype=torch.float64, requires_grad=False)
    fD = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=False)
    fW = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=False)
    fE = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=False)
    Throughfall = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=False)
    Interception = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=False)
    Snowmelt = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=False)
    Drainage = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=False)
    Canopywater = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=False)
    S = torch.tensor([0.]*leng, dtype=torch.float64, requires_grad=False)

    op = preles.preles(PAR=PAR, TAir = TAir, VPD = VPD, Precip = Precip, CO2 = CO2, fAPAR = fAPAR,GPPmeas = GPPmeas, ETmeas = ETmeas, SWmeas = SWmeas, GPP = GPP, ET = ET, SW = SW, SOG = SOG, fS = fS, fD = fD, fW = fW, fE = fE,Throughfall = Throughfall, Interception = Interception, Snowmelt = Snowmelt, Drainage = Drainage, Canopywater = Canopywater, S = S,soildepth = p1, ThetaFC = p2,  ThetaPWP = p3  , tauDrainage = p4  , beta = p5  , tau = p6  , S0= p7  , Smax = p8  , kappa= p9  ,gamma = p10  , soilthres = p11  , bCO2 = p12  , xCO2 = p13  , ETbeta = p14  , ETkappa = p15  , ETchi = p16  , ETsoilthres = p17  ,ETnu = p18  , MeltCoef = p19  , I0 = p20  , CWmax = p21  , SnowThreshold = p22  , T_0 = p23  , SWinit = p24  , CWinit = p25  ,SOGinit = p26  , Sinit = p27  , t0 = p28  , tcrit = p29  , tsumcrit = p30  , etmodel = control  , LOGFLAG = logflag, NofDays = NofDays  ,day = DOY  , transp = transp  , evap = evap  , fWE = fWE)


    GPP = op[0]
    ET = op[1]
    SW = op[2]
    tp=None
    if tp is None:
        out = GPP.unsqueeze(-1)
    else:
        out = orch.stack((GPP.flatten(), ET.flatten(), (SW.flatten()-sw[0])/sw[1]), dim=1)

    #print(op[0])
    return out
