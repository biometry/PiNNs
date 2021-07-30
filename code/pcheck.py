import utils
import trainloaded
import anomaly_detection
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
import preles
from torch import autograd


def physical_forward(input_data):#, mean, std):
        # extract Parameters
        p1 = torch.tensor(413.0, dtype=torch.float64) #torch.mean(parameters[..., 0:1], dtype=torch.float64)*400
        p2 = torch.tensor(0.45, dtype=torch.float64) #torch.mean(parameters[..., 1:2], dtype=torch.float64)
        p3 = torch.tensor(0.118, dtype=torch.float64) #torch.mean(parameters[..., 2:3], dtype=torch.float64)
        p4 = torch.tensor(3.0, dtype=torch.float64) #torch.mean(parameters[..., 3:4], dtype=torch.float64)
        p5 = torch.tensor(0.748018, dtype=torch.float64) #torch.mean(parameters[..., 4:5], dtype=torch.float64)
        p6 = torch.tensor(13.233830, dtype=torch.float64) #torch.mean(parameters[..., 5:6], dtype=torch.float64)*12
        p7 = torch.tensor(-3.965787, dtype=torch.float64) #torch.mean(parameters[..., 6:7], dtype=torch.float64)*10
        p8 = torch.tensor(18.766960, dtype=torch.float64) #torch.mean(parameters[..., 7:8], dtype=torch.float64)*10
        p9 = torch.tensor(-0.130473, dtype=torch.float64) #torch.mean(parameters[..., 8:9], dtype=torch.float64)
        p10 = torch.tensor(0.034459, dtype=torch.float64) #torch.mean(parameters[..., 9:10], dtype=torch.float64)
        p11 = torch.tensor(0.450828, dtype=torch.float64) #torch.mean(parameters[..., 10:11], dtype=torch.float64)
        p12 = torch.tensor(2000.0, dtype=torch.float64) #torch.mean(parameters[..., 11:12], dtype=torch.float64)
        p13 = torch.tensor(0.4, dtype=torch.float64) #torch.mean(parameters[..., 12:13], dtype=torch.float64)
        p14 = torch.tensor(0.324463, dtype=torch.float64) #torch.mean(parameters[..., 13:14], dtype=torch.float64)*10
        p15 = torch.tensor(0.874151, dtype=torch.float64) #torch.mean(parameters[..., 14:15], dtype=torch.float64)
        p16 = torch.tensor(0.075601, dtype=torch.float64) #torch.mean(parameters[..., 15:16], dtype=torch.float64)
        p17 = torch.tensor(0.541605, dtype=torch.float64) #torch.mean(parameters[..., 16:17], dtype=torch.float64)
        p18 = torch.tensor(0.273584, dtype=torch.float64) #torch.mean(parameters[..., 17:18], dtype=torch.float64)*5
        p19 = torch.tensor(1.2, dtype=torch.float64) #torch.mean(parameters[..., 18:19], dtype=torch.float64)
        p20 = torch.tensor(0.33, dtype=torch.float64) #torch.mean(parameters[..., 19:20], dtype=torch.float64)
        p21 = torch.tensor(4.970496, dtype=torch.float64) #torch.mean(parameters[..., 20:21], dtype=torch.float64)*4
        p22 = torch.tensor(0.0, dtype=torch.float64) #torch.mean(parameters[..., 21:22], dtype=torch.float64)
        p23 = torch.tensor(0.0, dtype=torch.float64) #torch.mean(parameters[..., 22:23], dtype=torch.float64)
        p24 = torch.tensor(200.0, dtype=torch.float64) #torch.mean(parameters[..., 23:24], dtype=torch.float64)*180
        p25 = torch.tensor(0.0, dtype=torch.float64) #torch.mean(parameters[..., 24:25], dtype=torch.float64)
        p26 = torch.tensor(0.0, dtype=torch.float64) #torch.mean(parameters[..., 25:26], dtype=torch.float64)
        p27 = torch.tensor(20.0, dtype=torch.float64) #torch.mean(parameters[..., 26:27], dtype=torch.float64)*10
        #p28 = torch.mean(parameters[..., 27:28], dtype=torch.float64)
        #p29 = torch.mean(parameters[..., 28:29], dtype=torch.float64)
        #p30 = torch.mean(parameters[..., 29:30], dtype=torch.float64)
        
        p28 = torch.tensor(-999., dtype=torch.float64)
        p29 = torch.tensor(-999., dtype=torch.float64)
        p30 = torch.tensor(-999., dtype=torch.float64)
        
        # Inputs
        
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
        print('Before Calc')
        op = preles.preles(PAR=PAR, TAir = TAir, VPD = VPD, Precip = Precip, CO2 = CO2, fAPAR = fAPAR, GPPmeas = GPPmeas, ETmeas = ETmeas, SWmeas = SWmeas, GPP = GPP, ET = ET, SW = SW, SOG = SOG, fS = fS, fD = fD, fW = fW, fE = fE, Throughfall = Throughfall, Interception = Interception, Snowmelt = Snowmelt, Drainage = Drainage, Canopywater = Canopywater, S = S, soildepth = p1, ThetaFC = p2,  ThetaPWP = p3  , tauDrainage = p4  , beta = p5  , tau = p6  , S0= p7  , Smax = p8  , kappa= p9  , gamma = p10  , soilthres = p11  , bCO2 = p12  , xCO2 = p13  , ETbeta = p14  , ETkappa = p15  , ETchi = p16  , ETsoilthres = p17  , ETnu = p18  , MeltCoef = p19  , I0 = p20  , CWmax = p21, SnowThreshold = p22  , T_0 = p23  , SWinit = p24  , CWinit = p25  , SOGinit = p26  , Sinit = p27  , t0 = p28  , tcrit = p29  , tsumcrit = p30  , etmodel = control  , LOGFLAG = logflag, NofDays = NofDays  , day = DOY  , transp = transp  , evap = evap  , fWE = fWE)
        print('after calc')
        GPP = op[0]#(op[0]-mean['GPP'])/std['GPP']
        ET = op[1]#(op[1]-mean['ET'])/std['ET']

        return GPP#torch.stack((GPP.type(torch.float32), ET.type(torch.float32)), dim=1)





with autograd.detect_anomaly():
        x, y, mn, std, xt = utils.loaddata('NAS', 0, dir="./data/", raw=True)
        xn = xt.drop(['date', 'GPP', 'ET'], axis=1)


        x_train, y_train = x, y
        xn_train = xn
        print(x_train.to_numpy())
        print(y_train.to_numpy())
        print(xn_train.to_numpy())
        train_set = TensorDataset(torch.tensor(x_train.to_numpy(), dtype=torch.float32), torch.tensor(y_train.to_numpy(), dtype=torch.float32), torch.tensor(xn_train.to_numpy(), dtype=torch.float32))
        train_set_size = len(train_set)
        sample_id = list(range(train_set_size))



        train_sampler = torch.utils.data.sampler.SequentialSampler(sample_id[:int(train_set_size // 100 * 80)])
        batchsize = 20     
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, sampler = train_sampler, shuffle=False)
                 
        for step, train_data in enumerate(train_loader):
                xt = train_data[0]
                yt = train_data[1]
                xnt = train_data[2]

                yhat = physical_forward(xnt)
                #crit = nn.L1Loss()
                #loss = crit(yhat, yt)
                yhat.sum().backward()
        
