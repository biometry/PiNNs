#include "prelesglobals.h"
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <stdio.h>

/* R interface function, replaces main() */
std::vector<torch::Tensor> call_preles(torch::Tensor *PAR, torch::Tensor *TAir, torch::Tensor *VPD, torch::Tensor *Precip, torch::Tensor *CO2,
		 torch::Tensor *fAPAR,
                 torch::Tensor *GPPmeas, torch::Tensor *ETmeas, torch::Tensor *SWmeas,
                 torch::Tensor *GPP, torch::Tensor *ET, torch::Tensor *SW, torch::Tensor *SOG,
                 torch::Tensor *fS, torch::Tensor *fD, torch::Tensor *fW,  torch::Tensor *fE,
                 torch::Tensor *Throughfall, torch::Tensor *Interception, torch::Tensor *Snowmelt,
                 torch::Tensor *Drainage,
                 torch::Tensor *Canopywater, torch::Tensor *S,
                 torch::Tensor *soildepth,
                 torch::Tensor *ThetaFC,
                 torch::Tensor *ThetaPWP,
                 torch::Tensor *tauDrainage,
                 torch::Tensor *beta,                                                             
                 torch::Tensor *tau,
                 torch::Tensor *S0,
                 torch::Tensor *Smax,
                 torch::Tensor *kappa,
                 torch::Tensor *gamma,
                 torch::Tensor *soilthres,                                                                      
                 torch::Tensor *bCO2,                                                                    
                 torch::Tensor *xCO2,                                                                  
                 torch::Tensor *ETbeta,                                              
                 torch::Tensor *ETkappa,
                 torch::Tensor *ETchi,
                 torch::Tensor *ETsoilthres,                                                                    
                 torch::Tensor *ETnu,                                                                
                 torch::Tensor *MeltCoef,                                                                   
                 torch::Tensor *I0,
                 torch::Tensor *CWmax,
                 torch::Tensor *SnowThreshold,
                 torch::Tensor *T_0,
                 torch::Tensor *SWinit,  
                 torch::Tensor *CWinit,                                                                 
                 torch::Tensor *SOGinit,
                 torch::Tensor *Sinit,                                                                           
                 torch::Tensor *t0,
                 torch::Tensor *tcrit,
                 torch::Tensor *tsumcrit,
		 int *etmodel, int *LOGFLAG, int *NofDays,
                 torch::Tensor *day,
                 torch::Tensor *transp,
                 torch::Tensor *evap,
                 torch::Tensor *fWE) {

  int preles(int NofDays, torch::Tensor *PAR, torch::Tensor *TAir, torch::Tensor *VPD, torch::Tensor *Precip,
		    torch::Tensor *CO2,
		    torch::Tensor *fAPAR,  p1 Site_par, p2 GPP_par, p3 ET_par, p4 SnowRain_par,
		    int etmodel,
		    torch::Tensor *GPP, torch::Tensor *ET, torch::Tensor *SW, torch::Tensor *SOG,
		    torch::Tensor *fS, torch::Tensor *fD, torch::Tensor *fW,  torch::Tensor *fE,
		    torch::Tensor *Throughfall, torch::Tensor *Interception, torch::Tensor *Snowmelt,
		    torch::Tensor *Drainage,
		    torch::Tensor *Canopywater,
		    torch::Tensor *GPPmeas, torch::Tensor *ETmeas, torch::Tensor *SWmeas, torch::Tensor *S,
	            int LOGFLAG, long int multisiteNday, 
		    torch::Tensor *day, 
		    torch::Tensor *transp, 
		    torch::Tensor *evap, torch::Tensor *fWE);

  std::cout << "co2check";
  std::cout << *CO2;
  
  /* Parameter structs */
  p1 parSite;
  p2 parGPP;
  p3 parET;
  p4 parSnowRain;
  
  /* Read in model parameters */
  parSite.soildepth = *soildepth;
  parSite.ThetaFC = *ThetaFC;
  parSite.ThetaPWP = *ThetaPWP;
  parSite.tauDrainage = *tauDrainage;
  parGPP.beta = *beta; 
  parGPP.tau = *tau;
  parGPP.S0 = *S0;
  parGPP.Smax = *Smax;
  parGPP.kappa = *kappa;
  parGPP.gamma = *gamma;
  parGPP.soilthres = *soilthres;
  parGPP.bCO2 = *bCO2;
  parGPP.xCO2 = *xCO2;
  parGPP.t0 = *t0;
  parGPP.tcrit = *tcrit;
  parGPP.tsumcrit = *tsumcrit;
  parET.beta = *ETbeta;
  parET.kappa = *ETkappa;
  parET.chi = *ETchi;
  parET.soilthres = *ETsoilthres;
  parET.nu = *ETnu;
  parSnowRain.MeltCoef = *MeltCoef;
  parSnowRain.I0 = *I0; 
  parSnowRain.CWmax = *CWmax;
  
  parSnowRain.SnowThreshold=torch::tensor(0., torch::requires_grad());
  parSnowRain.T_0=torch::tensor(0., torch::requires_grad());
  parSnowRain.SnowThreshold=torch::tensor(0., torch::requires_grad());
  parSnowRain.T_0=torch::tensor(0., torch::requires_grad());


  // Forward init values (previous day values) as first values of result vectors
  int n = *NofDays;
  torch::Tensor ini = torch::zeros(n);
  ini[0] = 1;
  ini.requires_grad_();
  torch::Tensor iniT = torch::ones(n);
  iniT[0] = 0;
  iniT.requires_grad_();
  
  *SW = *SW*iniT + *SWinit*ini;
  *Canopywater = *Canopywater*iniT + *CWinit*ini;
  *SOG = *SOG*iniT + *SOGinit*ini;
  *S = *S*iniT + *Sinit*ini;
  
  FILE *flog=NULL;
  if (*LOGFLAG > 0.5) {
    flog = fopen("preles.log", "w"); // EXCEPTION LOGGING
    if (flog) {
      fprintf(flog, "call_preles(): First day weather:\nDOY=", day[0], "\tPPFD=",PAR[0], "\tT=", TAir[0], "\tVPD=",VPD[0], "\tP=", Precip[0], "\tCO2=",CO2[0], "\nfAPAR=",
	      fAPAR[0], "\tSW=", SW[0], "\tCW=",Canopywater[0], "\tSOG=",SOG[0], "\tS=", S[0], "\n");
      if (*LOGFLAG > 1.5) 
	fprintf(flog, 
		"call_preles(): Parameters: N=",NofDays, "\tparGGP.beta=", parGPP.beta, "\tparET.chi=", parET.chi,"tparSnowRain.SnowThreshold=", parSnowRain.SnowThreshold, "\tetmodel=", etmodel, "\nLOGFLAG=", LOGFLAG, "parGPP.t0=", parGPP.t0, "\n");
      
    }  else {
      //exit(1);
    }
    fclose(flog);
  }
  
  /* Call the workhorse function ------------------------------------------ */
  int notinf;

  notinf = preles(*NofDays, PAR, TAir,
		  VPD, Precip,CO2,
		  fAPAR, parSite,
		  parGPP, parET, parSnowRain, *etmodel,
		  GPP, ET, SW, SOG, fS, fD, fW,  fE,
		  Throughfall, Interception, Snowmelt,
		  Drainage, Canopywater,
		  GPPmeas, ETmeas, SWmeas, S, *LOGFLAG, *NofDays, day, 
		  transp, evap, fWE);
                                
  if (*LOGFLAG > 0.5) {
    flog = fopen("preles.log", "a"); // EXCEPTION LOGGING 



    if (flog) {
      fprintf(flog,  "call_preles(): preles() returned code %d...finishing\n", notinf);
      fclose(flog);
    } else {
      //exit(1);
    }           
  }
  
  return { *GPP, *ET, *SW, *SOG, *fS, *fD, *fW, *fE, *Throughfall, *Interception, *Snowmelt, *Drainage, *Canopywater, *S};
};
namespace py = pybind11;

PYBIND11_MODULE(preles, m) {
  using namespace pybind11::literals;
  m.def("preles", &call_preles, "PAR"_a, "TAir"_a, "VPD"_a, "Precip"_a, "CO2"_a, "fAPAR"_a, "GPPmeas"_a, "ETmeas"_a, "SWmeas"_a, "GPP"_a, "ET"_a, "SW"_a,
	"SOG"_a, "fS"_a, "fD"_a, "fW"_a, "fE"_a, "Throughfall"_a, "Interception"_a, "Snowmelt"_a, "Drainage"_a, "Canopywater"_a, "S"_a, "soildepth"_a,
	"ThetaFC"_a, "ThetaPWP"_a, "tauDrainage"_a, "beta"_a, "tau"_a, "S0"_a, "Smax"_a, "kappa"_a, "gamma"_a, "soilthres"_a, "bCO2"_a, "xCO2"_a,
	"ETbeta"_a, "ETkappa"_a, "ETchi"_a, "ETsoilthres"_a, "ETnu"_a, "MeltCoef"_a, "I0"_a, "CWmax"_a, "SnowThreshold"_a, "T_0"_a, "SWinit"_a,
	"CWinit"_a, "SOGinit"_a, "Sinit"_a, "t0"_a, "tcrit"_a, "tsumcrit"_a, "etmodel"_a, "LOGFLAG"_a, "NofDays"_a, "day"_a, "transp"_a, "evap"_a, "fWE"_a);

  #ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIF(VERSION_INFO);
  #else
  m.attr("__version__") = "dev";
  #endif
}

