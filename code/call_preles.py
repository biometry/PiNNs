import torch
import preles
import utils
import numpy as np
import pandas as pd

def call(parameters, input_data, tp=None, sw=None):
    # extract Parameters
    p1 = parameters[0] 
    p2 = parameters[1]
    p3 = parameters[2]
    p4 = parameters[3]
    p5 = parameters[4]
    p6 = parameters[5]
    p7 = parameters[6]
    p8 = parameters[7]
    p9 = parameters[8]
    p10 = parameters[9]
    p11 = parameters[10]
    p12 = parameters[11]
    p13 = parameters[12]
    p14 = parameters[13]
    p15 = parameters[14]
    p16 = parameters[15]
    p17 = parameters[16]
    p18 = parameters[17]
    p19 = parameters[18]
    p20 = parameters[19]
    p21 = parameters[20]
    p22 = parameters[21]
    p23 = parameters[22]
    p24 = parameters[23]
    p25 = parameters[24]
    p26 = parameters[25]
    p27 = parameters[26]
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
    
    op = preles.preles(PAR=PAR, TAir = TAir, VPD = VPD, Precip = Precip, CO2 = CO2, fAPAR = fAPAR, GPPmeas = GPPmeas, ETmeas = ETmeas, SWmeas = SWmeas, GPP = GPP, ET = ET, SW = SW, SOG = SOG, fS = fS, fD = fD, fW = fW, fE = fE, Throughfall = Throughfall, Interception = Interception, Snowmelt = Snowmelt, Drainage = Drainage, Canopywater = Canopywater, S = S, soildepth = p1, ThetaFC = p2,  ThetaPWP = p3  , tauDrainage = p4  , beta = p5  , tau = p6  , S0= p7  , Smax = p8  , kappa= p9  , gamma = p10  , soilthres = p11  , bCO2 = p12  , xCO2 = p13  , ETbeta = p14  , ETkappa = p15  , ETchi = p16  , ETsoilthres = p17  , ETnu = p18  , MeltCoef = p19  , I0 = p20  , CWmax = p21  , SnowThreshold = p22  , T_0 = p23  , SWinit = p24  , CWinit = p25  , SOGinit = p26  , Sinit = p27  , t0 = p28  , tcrit = p29  , tsumcrit = p30  , etmodel = control  , LOGFLAG = logflag, NofDays = NofDays  , day = DOY  , transp = transp  , evap = evap  , fWE = fWE)


    GPP = op[0]
    ET = op[1]
    SW = op[2]
    if tp is None:
        out = GPP.unsqueeze(-1)
    else:
        out = torch.stack((GPP.flatten(), ET.flatten(), (SW.flatten()-sw[0])/sw[1]), dim=1)
        
    
    return out


xx, xy, yt = utils.loaddata('NAS', 0, dir='./data/', raw=True)


yt.index = pd.DatetimeIndex(yt['date'])
x = yt[['Precip', 'Tair', 'PAR', 'VPD', 'fapar', 'DOY', 'CO2']]
y = yt['GPP']

test_x = x[x.index.year == 2004]
test_y = y[y.index.year == 2004]
#train_x.index, train_y.index = np.arange(0, len(train_x)), np.arange(0, len(train_y)) 
test_x.index, test_y.index = np.arange(0, len(test_x)), np.arange(0, len(test_y))

#test_x = test_x.to_frame()
test_y = test_y.to_frame()
test_x, test_y = torch.tensor(test_x.to_numpy(), dtype=torch.float32), torch.tensor(test_y.to_numpy(), dtype=torch.float32)

parameters = [torch.tensor(413, dtype=torch.float32),
              torch.tensor(0.45, dtype=torch.float32),
              torch.tensor(0.118, dtype=torch.float32),
              torch.tensor(3, dtype=torch.float32),
              torch.tensor(0.748018, dtype=torch.float32),
              torch.tensor(13.23383, dtype=torch.float32),
              torch.tensor(-3.9657867, dtype=torch.float32),
              torch.tensor(18.76696, dtype=torch.float32),
              torch.tensor(-0.130473, dtype=torch.float32),
              torch.tensor(0.034459, dtype=torch.float32),
              torch.tensor(0.450828, dtype=torch.float32),
              torch.tensor(2000, dtype=torch.float32),
              torch.tensor(0.4, dtype=torch.float32),
              torch.tensor(0.324463, dtype=torch.float32),
              torch.tensor(0.874151, dtype=torch.float32),
              torch.tensor(0.075601, dtype=torch.float32),
              torch.tensor(0.541605, dtype=torch.float32),
              torch.tensor(0.273584, dtype=torch.float32),
              torch.tensor(1.2, dtype=torch.float32),
              torch.tensor(0.33, dtype=torch.float32),
              torch.tensor(4.970496, dtype=torch.float32),
              torch.tensor(0, dtype=torch.float32),
              torch.tensor(0, dtype=torch.float32),
              torch.tensor(200, dtype=torch.float32),
              torch.tensor(0, dtype=torch.float32),
              torch.tensor(0, dtype=torch.float32),
              torch.tensor(20, dtype=torch.float32)]
              #torch.tensor(-999, dtype=torch.float32),
              #torch.tensor(-999, dtype=torch.float32),
              #torch.tensor(-999, dtype=torch.float32),
              #torch.tensor(1, dtype=torch.float32),
              #torch.tensor(1, dtype=torch.float32)]

print(len(parameters))
op = call(parameters, test_x)
print(op)
mae = (test_y - op.flatten())
print(torch.mean(mae))
