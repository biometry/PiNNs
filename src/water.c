#include "prelesglobals.h"
#include <pybind11/pybind11.h>
#include <TH/TH.h>
#include <stdio.h>

/* Estimate Evapotranspiration according to a simple empirical model
 * that uses GPP prediction to calculate transpiration, as driven
 * by VPD. Evaporation is estimated with PPFD, which is a surrogate
 * for Rnet */
torch::Tensor ETfun(torch::Tensor D, torch::Tensor theta, torch::Tensor ppfd, torch::Tensor fAPAR, torch::Tensor T, 
		    p3 ET_par, p1 Site_par,
		    torch::Tensor *canw,
		    torch::Tensor *fE, torch::Tensor A,
		    torch::Tensor fWgpp,  p2 GPP_par,  //double fCO2mean, 
		    torch::Tensor CO2, 
		    FILE *flog, int LOGFLAG, int etmodel, torch::Tensor *transp, 
		    torch::Tensor *evap, torch::Tensor *fWE, int i, torch::Tensor Msk, torch::Tensor msk) {
  extern torch::Tensor fCO2_ET_model_mean(torch::Tensor CO2, p2 GPP_par, torch::Tensor Msk );
  //torch::Tensor pow();

  
  torch::Tensor thetavol = theta/Site_par.soildepth; 
  torch::Tensor REW=(thetavol-Site_par.ThetaPWP)/
    (Site_par.ThetaFC-Site_par.ThetaPWP);
  //  double fEsub = -999; /* Minimum of fW and fD returned if ET-model 
  //			* flag indicates similar modifier as for GPP */
  torch::Tensor fWsub=torch::tensor(1.0, torch::requires_grad());
  //  double fDsub=1;
  torch::Tensor et; 
  torch::Tensor fCO2mean;
  torch::Tensor lambda, psychom, s; //, rho;
  torch::Tensor cp = torch::tensor(1003.5, torch::requires_grad()); // J/(kg K) (nearly constant, this is dry air on sea level)
  torch::Tensor MWratio = torch::tensor(0.622, torch::requires_grad()); // Ratio of molecular weigths of water vapor and dry air;
  // double R = 287.058; // J/(kg K) Specific gas constant for dry air, wiki
  // double zh, zm, d, zom, zoh;
  /*If pressure is not inputted use default */
  torch::Tensor pressure = torch::tensor(101300., torch::requires_grad()); // Pa  
  fCO2mean = fCO2_ET_model_mean(CO2, GPP_par, Msk);

  // rho=pressure/(R * (T+273.15) ); // Dry air density, kg/m3
  lambda = ((-0.0000614342 * T.pow(3) + 0.00158927 * T.pow(2) - 
	     2.36418 * T +  2500.79) * 1000)*Msk; // J/kg
  psychom = ((cp * pressure).pow(Msk) /(lambda * MWratio).pow(Msk))*Msk; // Pa/C, wiki
  s = (1000 * 4098.0 * (0.6109 * torch::exp((17.27 * T)/(T+237.3))) / 
       (T+237.3).pow(2))*Msk;  // Pa/C! (Ice has nearly the same slope)
  
  /* Calculate soil constraint, simple linear following Granier 1987*/
  if (torch::lt(ET_par.soilthres, -998).item<bool>()) { /*-999 omits water control*/
    fWsub = Msk; 
  } else {
    if (torch::lt(REW, ET_par.soilthres).item<bool>()) {
      if (torch::gt(REW, 0.01).item<bool>()) fWsub = (REW/ET_par.soilthres)*Msk; else fWsub = torch::zeros(Msk.sizes(), torch::requires_grad());
    } else {
      fWsub = Msk;
    }
  }
  
  /* If there is any water in canopy, evaporation is not reduced by
   * low soil water */
  
  if (torch::gt(*canw, 0.00000001).item<bool>()){
    fWsub = Msk;
  }
  //  if (fDsub > fWsub) fEsub=fWsub; else fEsub = fDsub;     
  *fE = *fE*msk + fWsub;   
  *fWE = *fWE*msk + fWsub;

  if (torch::any(torch::lt(D*Msk, 0.01*Msk)).item<bool>()) {
    D = D*msk + Msk*torch::tensor(0.01, torch::requires_grad());
  }
  if (LOGFLAG > 1.5)
    fprintf(flog, "   ETfun(): CO2mean=%lf\tat CO2=%lf\n", 
	    fCO2mean, CO2);
  
  if (etmodel == -1) {
    if ((torch::lt(torch::nansum(D*Msk), 0).item<bool>() & (torch::lt(ET_par.kappa, 0).item<bool>() & torch::gt(ET_par.kappa, -1).item<bool>()) | (torch::gt(ET_par.kappa, 0).item<bool>() & torch::lt(ET_par.kappa, 1).item<bool>())) | (torch::lt(torch::nansum(fWgpp*Msk), 0).item<bool>() & (torch::lt(ET_par.nu, 0).item<bool>() & torch::gt(ET_par.nu, -1).item<bool>()) | (torch::gt(ET_par.nu, 0).item<bool>() & torch::lt(ET_par.nu, 1).item<bool>()))){
      *transp = *transp;
	 
	 } else {  
   
	
    *transp = *transp*msk + (D * ET_par.beta*A/(D*Msk).pow(ET_par.kappa) *
			     (fWgpp*Msk).pow(ET_par.nu) * // ET differently sensitive to soil water than GPP
			     fCO2mean)*Msk;
    }
    *evap = *evap*msk + (ET_par.chi *  (1-fAPAR) *  fWsub * ppfd)*Msk;
    et = (*transp + *evap * s / (s + psychom))*Msk; 
  }
  if (etmodel == 0) {
    if ((torch::lt(torch::nansum(D*Msk), 0).item<bool>() & (torch::lt(ET_par.kappa, 0).item<bool>() & torch::gt(ET_par.kappa, -1).item<bool>()) | (torch::gt(ET_par.kappa, 0).item<bool>() & torch::lt(ET_par.kappa, 1).item<bool>())) | (torch::lt(torch::nansum(fWgpp*Msk), 0).item<bool>() & (torch::lt(ET_par.nu, 0).item<bool>() & torch::gt(ET_par.nu, -1).item<bool>()) | (torch::gt(ET_par.nu, 0).item<bool>() & torch::lt(ET_par.nu, 1).item<bool>()))){
      *transp = *transp;

       } else {
      *transp = *transp*msk + (Msk * D * ET_par.beta * A/(D*Msk).pow(ET_par.kappa) *
			       (fWgpp*Msk).pow(ET_par.nu) * // ET differently sensitive to soil water than GPP
			     fCO2mean)*Msk;
    }
    *evap = *evap*msk + ((ET_par.chi *  s).pow(Msk) / (s + psychom).pow(Msk) * (1-fAPAR) *  fWsub * ppfd)*Msk;
    //    et = D * ET_par.beta*A/pow(D, ET_par.kappa) *
    //  pow(fWgpp, ET_par.nu) * // ET differently sensitive to soil water than GPP
    //  fCO2mean +  // Mean effect of CO2 on transpiration
    //  ET_par.chi *  s / (s + psychom) * (1-fAPAR) *  fWsub * ppfd;

    et = *transp*Msk + *evap*Msk;
  }
  if (etmodel == 1) {
    if ((torch::lt(torch::nansum(D*Msk), 0).item<bool>() & (torch::lt(ET_par.kappa, 0).item<bool>() & torch::gt(ET_par.kappa, -1).item<bool>()) | (torch::gt(ET_par.kappa, 0).item<bool>() & torch::lt(ET_par.kappa, 1).item<bool>())) | (torch::lt(torch::nansum(fWgpp*Msk), 0).item<bool>() & (torch::lt(ET_par.nu, 0).item<bool>() & torch::gt(ET_par.nu, -1).item<bool>()) | (torch::gt(ET_par.nu, 0).item<bool>() & torch::lt(ET_par.nu, 1).item<bool>()))){
      *transp = *transp;

      } else {
      *transp = *transp*msk + (D * ET_par.beta*A/(D*Msk).pow(ET_par.kappa) *
			       (fWgpp*Msk).pow(ET_par.nu) * // ET differently sensitive to soil water than GPP
			     fCO2mean)*Msk;
    }
    *evap = *evap*msk + (ET_par.chi * (1-fAPAR) *  fWsub * ppfd)*Msk;
    //et = D * ET_par.beta*A/pow(D, ET_par.kappa) *
    //  pow(fWgpp, ET_par.nu) * // ET differently sensitive to soil water than GPP
    //  fCO2mean +  // Mean effect of CO2 on transpiration
    //  ET_par.chi * (1-fAPAR) *  fWsub * ppfd;
    et = (*transp + *evap)*Msk;
  }
  
  if (etmodel == 2) {
    if ((torch::lt(torch::nansum(D*Msk), 0).item<bool>() & (torch::lt(ET_par.kappa, 0).item<bool>() & torch::gt(ET_par.kappa, -1).item<bool>()) | (torch::gt(ET_par.kappa, 0).item<bool>() & torch::lt(ET_par.kappa, 1).item<bool>())) | (torch::lt(torch::nansum(fWgpp*Msk), 0).item<bool>() & (torch::lt(ET_par.nu, 0).item<bool>() & torch::gt(ET_par.nu, -1).item<bool>()) | (torch::gt(ET_par.nu, 0).item<bool>() & torch::lt(ET_par.nu, 1).item<bool>()))){
      et = (ET_par.chi * (1-fAPAR) * fWsub * ppfd)*Msk;

       } else {
         et = (D * (1 + ET_par.beta/(D*Msk).pow(ET_par.kappa)) * A / CO2 * 
	   (fWgpp*Msk).pow(ET_par.nu) * // ET differently sensitive to soil water than GPP
	   fCO2mean +  // Mean effect of CO2 on transpiration      
	    ET_par.chi * (1-fAPAR) *  fWsub * ppfd)*Msk;
    }
  }
  
  
  if (LOGFLAG > 2.5)
    fprintf(flog, "      ETfun(): Model=%d\nD\t%lf\nET_par.beta\t%lf\nA\t%lf\npow(D, ET_par.kappa)\t%lf\npow(fWgpp, ET_par.nu)\t%lf\nfWgpp\t%lf\nET_par.nu\t%lf\nfCO2mean\t%lf\nCO2\t%lf\nET_par.chi\t%lf\ns/(s+psychom)\t%lf\n1-fAPAR\t%lf\nfWsum\t%lf\nppfd\t%lf\n-->et\t%lf\n",	    
	    etmodel, D, ET_par.beta, A, D.pow(ET_par.kappa), 
	    fWgpp.pow(ET_par.nu), fWgpp, ET_par.nu,
	    fCO2mean, 
	    CO2,
	    ET_par.chi ,   s / (s + psychom), 1-fAPAR, fWsub,  ppfd, et);

  
  return(et);
}


/*Interception is a fraction of daily rainfall, fraction depending on fAPAR*/
void  interceptionfun(torch::Tensor *rain, torch::Tensor *intercepted, torch::Tensor Temp, p4 
		      SnowRain_par, torch::Tensor fAPAR, int i, torch::Tensor Msk, torch::Tensor msk) {
  if (torch::any(torch::gt(Temp*Msk, SnowRain_par.SnowThreshold*Msk)).item<bool>())  {
    *intercepted = *intercepted*msk + *rain*Msk * ((SnowRain_par.I0 * fAPAR) / 0.75); 
    *rain = *rain*msk + *rain*Msk - *intercepted;
  } else {
    *intercepted = *intercepted*msk;
  }
}




/* Soil water balance is updated with snowmelt and canopy throughfall
 * and evapotranspiration. No drainage occurs below field capacity */
void swbalance(torch::Tensor *theta, torch::Tensor throughfall, torch::Tensor snowmelt, torch::Tensor et, 
               p1 sitepar, torch::Tensor *drainage,
	       torch::Tensor *snow, torch::Tensor *canw, p4 SnowRain_par, int i, torch::Tensor Msk, torch::Tensor msk) {
  torch::Tensor st0, etfromvegandsoil=torch::zeros(Msk.sizes(), torch::requires_grad());
  /* Evaporate first from wet canopy and snow on ground */
  if (torch::gt(SnowRain_par.CWmax, 0.00000001).item<bool>()) {
    if (torch::any(torch::gt((*canw*Msk + *snow*Msk -  et*Msk), 0*Msk)).item<bool>()) {
      if (torch::any(torch::gt((*canw*Msk - et*Msk), 0*Msk)).item<bool>()) { 
	*canw = torch::nansum(*canw*Msk - et*Msk);
	etfromvegandsoil = torch::tensor(0., torch::requires_grad());
      } else if (torch::any(torch::lt((*canw*Msk - et*Msk), 0*Msk)).item<bool>()) { // in this case, there's enough snow left
	*snow = torch::nansum(*snow*Msk + *canw*Msk - et*Msk);
	*canw = torch::tensor(0., torch::requires_grad());
	etfromvegandsoil = torch::zeros(Msk.sizes(), torch::requires_grad());
      }    
    } else {
      etfromvegandsoil = et*Msk - *canw*Msk - *snow*Msk;
      *canw= torch::tensor(0., torch::requires_grad());
      *snow = torch::tensor(0., torch::requires_grad());
    }

  } else {
    if (torch::any(torch::gt((*snow*Msk - et*Msk), 0*Msk)).item<bool>()) {
      *snow = torch::nansum(*snow*Msk - et*Msk);
      etfromvegandsoil = torch::zeros(Msk.sizes(), torch::requires_grad());
    } else if (torch::any(torch::lt((*snow*Msk - et*Msk), 0*Msk)).item<bool>()) { // in this case, there's enough snow left
      etfromvegandsoil = et*Msk - *snow*Msk;
      *snow = torch::tensor(0., torch::requires_grad());
    } else {
      *snow = torch::tensor(0., torch::requires_grad());
    }
  }
  
  et = etfromvegandsoil;
  /*  balance without drainage */
  st0 = (*theta + throughfall + snowmelt - et)*Msk;
  
  if (torch::any(torch::le(st0.pow(Msk), 0)).item<bool>()){
    
    st0 = Msk*torch::tensor(0.0001, torch::requires_grad());
  }
  /* Calculate what is left to drainage after partial balance update above: */
  if (torch::gt(sitepar.tauDrainage, 0).item<bool>()) {  
    // Simple time delay drainage above FC:
    if (torch::any(torch::gt(st0*Msk, (sitepar.ThetaFC * sitepar.soildepth)*Msk)).item<bool>()) { 
      *drainage = *drainage*msk + ((st0 - sitepar.ThetaFC * sitepar.soildepth) / 
				   sitepar.tauDrainage)*Msk;      
    } else {
      *drainage = *drainage*msk;
    }
    
    *theta = torch::nansum(st0*Msk - *drainage*Msk);


    /* Include marginal drainage below FC.
     * This was needed for model calibration only, below FC drainage
     * was practically zero, but important for convergence */
    /*
if (st0 > sitepar.ThetaFC * sitepar.soildepth) {
      *drainage = (st0 - sitepar.ThetaFC * sitepar.soildepth) / 
	sitepar.tauDrainage;      
    }
    if (*drainage < (sitepar.ThetaFC * sitepar.soildepth - 
		     sitepar.ThetaPWP * sitepar.soildepth) / 
	10000) //pow(sitepar.tauDrainage, 5)) 
	*drainage = (sitepar.ThetaFC * sitepar.soildepth - 
		     sitepar.ThetaPWP * sitepar.soildepth) / 
	  10000; //pow(sitepar.tauDrainage, 5);
    
    if (st0 <= sitepar.ThetaFC * sitepar.soildepth && 
	st0 > sitepar.ThetaPWP * sitepar.soildepth) { 
      *drainage = (st0 - sitepar.ThetaPWP * sitepar.soildepth) / 
	10000; //pow(sitepar.tauDrainage, 5);      
      *theta = st0 - *drainage;
    }
  
    if (st0 <= sitepar.ThetaPWP * sitepar.soildepth) {
      *drainage = 0;
      *theta = st0;
    }
    *theta = st0 - *drainage;
    */
    //****************************************************** */
      

  } 
  
}


/* Rain is snow below T > 0 C, and snow melts above O C. */
void  Snow(torch::Tensor T, torch::Tensor *rain, torch::Tensor *snow, p4 SnowRain_par, 
	   torch::Tensor *SnowMelt, int i, torch::Tensor Msk, torch::Tensor msk) {
  torch::Tensor NewSnow;
  
  if (torch::any(torch::lt(T*Msk, SnowRain_par.SnowThreshold*Msk)).item<bool>()) {
    NewSnow = *rain*Msk; 
    *rain = *rain * msk ; 
  } else {
    NewSnow=torch::zeros(Msk.sizes(), torch::requires_grad());
  } 
  
  if (torch::any(torch::gt(T*Msk, SnowRain_par.T_0*Msk)).item<bool>()){ 
    *SnowMelt = *SnowMelt*msk + SnowRain_par.MeltCoef.pow(Msk)*(T*Msk-(SnowRain_par.T_0*Msk));  
  } else {
    *SnowMelt= *SnowMelt*msk;
  
  }
  if (torch::any(torch::lt((*snow*Msk + NewSnow*Msk - *SnowMelt*Msk), 0*Msk)).item<bool>()) {
    *SnowMelt = *SnowMelt*msk + NewSnow + *snow*Msk;
    *snow = torch::tensor(0., torch::requires_grad());
  } else {
    *snow = torch::nansum((*snow + NewSnow - *SnowMelt)*Msk);
    
  }
  
};  
