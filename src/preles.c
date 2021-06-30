#include "prelesglobals.h"
#include <torch/extension.h>
#include <stdio.h>

using namespace torch::indexing;

int preles(int NofDays,
	   torch::Tensor *PAR, torch::Tensor *TAir, torch::Tensor *VPD, torch::Tensor *Precip, 
	   torch::Tensor *CO2,
	   torch::Tensor *fAPAR,  p1 Site_par,
		     p2 GPP_par, p3 ET_par,  p4 SnowRain_par, int etmodel,
	   torch::Tensor *GPP, torch::Tensor *ET, torch::Tensor *SW, torch::Tensor *SOG,
	   torch::Tensor *fS, torch::Tensor *fD, torch::Tensor *fW,  torch::Tensor *fE,
	   torch::Tensor *Throughfall, torch::Tensor *Interception, torch::Tensor *Snowmelt,
	   torch::Tensor *Drainage, torch::Tensor *Canopywater,
	   torch::Tensor *GPPmeas, torch::Tensor *ETmeas, torch::Tensor *SWmeas, torch::Tensor *S, 	   
	   int LOGFLAG, long int multisiteNday, torch::Tensor *day, 
	   torch::Tensor *transp, 
	   torch::Tensor *evap, torch::Tensor *fWE){

  extern torch::Tensor fS_model(torch::Tensor *S, torch::Tensor T, p2 GPP_par, torch::Tensor Msk, int i);
  extern torch::Tensor fPheno_model(p2 GPP_par, torch::Tensor T, torch::Tensor *PhenoS, 
				    torch::Tensor DOY, torch::Tensor fS, torch::Tensor Msk, int i, torch::Tensor msk);
    


  extern torch::Tensor ETfun(torch::Tensor D, torch::Tensor theta, torch::Tensor ppfd, torch::Tensor fAPAR, 
		      torch::Tensor T,
		      p3 ET_par, p1 Site_par,
		      torch::Tensor *canw,
		      torch::Tensor *fE, torch::Tensor A,
		      torch::Tensor fWgpp, p2 GPP_par,  //double fCO2mean, 
		      torch::Tensor CO2, 
		      FILE *flog, int LOGFLAG, int etmodel, 
		      torch::Tensor *transp, 
		      torch::Tensor *evap, torch::Tensor *fWE, int i, torch::Tensor Msk, torch::Tensor msk);
  
  extern void  interceptionfun(torch::Tensor *rain, torch::Tensor *intercepted, torch::Tensor Temp, p4
			       SnowRain_par, torch::Tensor fAPAR, int i, torch::Tensor Msk, torch::Tensor msk);
  extern void swbalance(torch::Tensor *theta, torch::Tensor throughfall, 
			torch::Tensor snowmelt, torch::Tensor et,
			p1 sitepar, torch::Tensor *drainage,
			torch::Tensor *snow, torch::Tensor *canw, p4 SnowRain_par, int i, torch::Tensor Mask, torch::Tensor mask);
  extern void  Snow(torch::Tensor T, torch::Tensor *rain, torch::Tensor *snow, p4 SnowRain_par,
		    torch::Tensor *SnowMelt, int i, torch::Tensor Msk, torch::Tensor msk);
  extern void initConditions(torch::Tensor **PAR, torch::Tensor **TAir, 
			     torch::Tensor **VPD, torch::Tensor **Precip,
			     torch::Tensor **CO2, torch::Tensor Msk, torch::Tensor msk);

  extern void GPPfun(torch::Tensor *gpp, torch::Tensor *gpp380, torch::Tensor ppfd,  
		     torch::Tensor D, torch::Tensor CO2, 
		     torch::Tensor theta, 
		     torch::Tensor fAPAR, torch::Tensor fSsub, 
		     p2 GPP_par, p1 Site_par, torch::Tensor *fD, torch::Tensor *fW,
	             torch::Tensor *fE, FILE *flog, int LOGFLAG, int i, torch::Tensor Msk, torch::Tensor msk);
    
  //extern double fCO2_VPD_exponent(double CO2, double xCO2 ) ;

  
  
  //extern double fCO2_ET_VPD_correction(double fEgpp, double xCO2 );
  //extern double fCO2_model_mean(double CO2, double bCO2 ) ;

  FILE *flog=NULL;
  flog = fopen("preles.log", "a"); // EXCEPTION LOGGING
  
  if (LOGFLAG > 0.5) fprintf(flog, "  Stepped into preles()");
  
  torch::Tensor theta;
  torch::Tensor theta_snow;
  torch::Tensor theta_canopy;
  torch::Tensor S_state;
  torch::Tensor PhenoS=torch::tensor(0., torch::requires_grad());
  torch::Tensor fPheno=torch::tensor(0., torch::requires_grad());
  int i; 
  torch::Tensor fEgpp = torch::tensor(0., torch::requires_grad());
  torch::Tensor gpp380 = torch::tensor(0., torch::requires_grad());
  
  int n = NofDays;
  torch::Tensor Mini = torch::zeros(n);
  Mini[0] = 1;
  Mini.requires_grad_();
 
  torch::Tensor mini = torch::ones(n);
  mini[0] = 0;
  mini.requires_grad_();
  std::cout << "-1CO2";
  std::cout << *CO2;
  initConditions(&PAR, &TAir, &VPD, &Precip, &CO2, Mini, mini);
  theta = torch::nansum(*SW*Mini);
  theta_canopy = torch::nansum(*Canopywater*Mini);
  theta_snow = torch::nansum(*SOG*Mini);
  S_state = torch::nansum(*S*Mini);
  std::cout << "thetaINI";
  std::cout << theta;

  if (LOGFLAG > 1.5) {
    fprintf(flog, "   preles(): Starting values for storage components:\nSW=", theta, "\tCW=", theta_canopy, "\tSOG=", theta_snow, "\tS=", S[0], "\n",
	    "   ...will loop %d rows of weather input\n");
    printf("   preles(): Site fAPAR =", fAPAR[0], " LUE =", GPP_par.beta, "and soil depth =", Site_par.soildepth, "\n");
  }

  
  //    fclose(flog);
  
  
  /*------- LOOPING DAYS---------------------------------------------------*/
  /* ---------------- ----------------------------------------------------*/  
  //int d = NofDays.item<int>();
  for (i=0; i < n; i++) { 
    torch::Tensor mask = torch::ones(n);
    mask[i] = 0;
    mask.requires_grad_();

    torch::Tensor Mask = torch::zeros(n);
    Mask[i] = 1;
    Mask.requires_grad_();
    

    if ((LOGFLAG > 1.5)) {
      fprintf(flog, "   \ni=", i+1, NofDays, "\t SW=", theta, "\tCW=", theta_canopy, "\tSOG=", theta_snow, "\tS=",S_state, "\n");
    }      
    std::cout << "i";
    std::cout << i;
    /* Use previous day environment for prediction, if current values are missing,
       or suspicious*/
    if (i > 0) {      
      std::cout << "build masks";
      torch::Tensor bmmask = torch::zeros(n);
      bmmask[i-1] = 1;
      bmmask.requires_grad_();

      
      torch::Tensor par = torch::zeros(n);
      par[i] = torch::nansum(*PAR*bmmask);
      par.requires_grad_();

      torch::Tensor tair = torch::zeros(n);
      tair[i] = torch::nansum(*TAir*bmmask);
      tair.requires_grad_();

      torch::Tensor vpd = torch::zeros(n);
      vpd[i] = torch::nansum(*VPD*bmmask);
      vpd.requires_grad_();

      torch::Tensor precip = torch::zeros(n);
      precip[i] = torch::nansum(*Precip*bmmask);
      precip.requires_grad_();

      torch::Tensor co2 = torch::zeros(n);
      co2[i] = torch::nansum(*CO2*bmmask);
      co2.requires_grad_();

      torch::Tensor gppmeas = torch::zeros(n);
      gppmeas[i] = torch::nansum(*GPPmeas*bmmask);
      gppmeas.requires_grad_();

      torch::Tensor etmeas = torch::zeros(n);
      etmeas[i] = torch::nansum(*ETmeas*bmmask);
      etmeas.requires_grad_();

      torch::Tensor swmeas = torch::zeros(n);
      swmeas[i] = torch::nansum(*SWmeas*bmmask);
      swmeas.requires_grad_();

      torch::Tensor sw = torch::zeros(n);
      sw[i] = torch::nansum(*SW*bmmask);
      sw.requires_grad_();

      torch::Tensor sog = torch::zeros(n);
      sog[i] = torch::nansum(*SOG*bmmask);
      sog.requires_grad_();
      
      if (torch::any(torch::lt(*PAR*Mask, -900)).item<bool>()) *PAR = *PAR * mask + par;
      if (torch::any(torch::lt(*TAir*Mask, -900)).item<bool>()) *TAir = *TAir * mask + tair;
      if (torch::any(torch::lt(*VPD*Mask, 0)).item<bool>() || torch::any(torch::gt(*VPD*Mask, 6)).item<bool>())  *VPD = *VPD * mask + tair;
      if (torch::any(torch::lt(*Precip*Mask, 0)).item<bool>()) *Precip = *Precip * mask + (precip * 0.3); 
      /* On avg. P+1=0.315*P 
	 * (in Sodis & Hyde) */
      if (torch::any(torch::lt(*CO2*Mask, 0)).item<bool>()) *CO2 = *CO2 * mask + co2;
      if (torch::any(torch::lt(*GPPmeas*Mask, -990)).item<bool>()) *GPPmeas = *GPPmeas * mask + gppmeas; 
      if (torch::any(torch::lt(*ETmeas*Mask, -990)).item<bool>()) *ETmeas = *ETmeas * mask + etmeas;    
      if (torch::any(torch::lt(*SWmeas*Mask, 0.0)).item<bool>()) *SWmeas = *SWmeas * mask + swmeas;   
      if (torch::any(torch::lt(*SW*Mask, -900)).item<bool>()) *SW = *SW * mask + sw; 
      if (torch::any(torch::lt(*SOG*Mask, -900)).item<bool>()) *SOG = *SOG * mask + sog; // See above, could be used for 
      
      	
    }

    if ((LOGFLAG > 1.5)) {
      fprintf(flog, "   weather inputs: PAR, T, VPD, P, CO2",
	      PAR[i], TAir[i], VPD[i], Precip[i], CO2[i]);
    }   

  
    /* Update temperature state that tells about seasonality -
     * for GPP and through GPP to ET */
    
    *fS = *fS*mask + fS_model(&S_state, *TAir, GPP_par, Mask, i);
    
    if (LOGFLAG > 1.5) fprintf(flog, "   preles(): estimated fS=\n", fS[i]);
    
    
    /* Deciduous phenology - don't use if this information is inputted in fAPAR */
    /* Note also that fapar is multiplied by 0 or 1 (i.e. leaf development is not accounted for) */
    /* Model predicts budbreak based on critical threshold temperature sum */ 
    /* Note that this implementation works only if data starts before t0-date of fPheno-model */      
    if (LOGFLAG > 1.5) fprintf(flog, 
			       "   preles(): stepping into fPheno_model: inputs:\n      GPP_par.t0, T, PhenoS, DOY", 
			       GPP_par.t0, TAir, PhenoS, day[i]);
    std::cout << *day;
    torch::Tensor DAY = torch::nansum(*day*Mask);
    
    fPheno = fPheno_model(GPP_par, *TAir, &PhenoS, DAY, *fS, Mask, i, mask);
    
    if (LOGFLAG > 1.5) fprintf(flog, "   preles(): PhenoS, fPheno", 
			       PhenoS, fPheno );

    *fAPAR = *fAPAR*mask + *fAPAR*Mask * fPheno; 

    if (LOGFLAG > 1.5) fprintf(flog, 
			       "   preles(): fAPAR changed to \n", 
			       fAPAR[i]);
    
    GPPfun(&GPP[0], &gpp380, *PAR, *VPD, *CO2, theta, *fAPAR, *fS,
		    GPP_par, Site_par,  &fD[0], &fW[0], &fEgpp, 
	   flog, LOGFLAG, i, Mask, mask);
    
    
    if (LOGFLAG > 1.5) 
      fprintf(flog, 
	      "   preles(): estimated GPP, fD, fEgg, GPP380ppm\n", 
	      GPP[i], fD[i], fEgpp, gpp380);
    
    std::cout << "TS_bSnow";
    std::cout << theta_snow;
    /* Calculate amount of snow and snowmelt at the end of the day */
    Snow(*TAir, &Precip[0], &theta_snow, SnowRain_par, &Snowmelt[0], i, Mask, mask);
    // NOTE: interception model could be better
    std::cout << theta_snow;
    
    *Throughfall  = *Throughfall*mask + *Precip*Mask;
    
    interceptionfun(&Throughfall[0], &Interception[0], *TAir, 
		    SnowRain_par, *fAPAR, i, Mask, mask);
    
    //if (LOGFLAG > 1.5) 
      //fprintf(flog, 
      //      "   preles(): estimated Thr.fall, Intercept, SOG, Snowmelt", 
      //      Throughfall->index({i}), Interception[i], SOG->index({i}), Snowmelt->index({i}));
    
    /*Excess water from canopy will drip down to soil if not evaporated 
      during the day, rest remains in canopy for the next day*/
    std::cout << "bIFtcanop";
    std::cout << theta_canopy;
    if (torch::lt(SnowRain_par.CWmax, 0.000000011).item<bool>()) { 
      *Throughfall = *Throughfall + *Interception*Mask;
    } else {
      if (torch::any(torch::lt((SnowRain_par.CWmax* *fAPAR)*Mask, (*Interception + theta_canopy)*Mask)).item<bool>()) {
	*Throughfall = *Throughfall + *Interception*Mask + theta_canopy*Mask - (SnowRain_par.CWmax* *fAPAR)*Mask;
	theta_canopy = torch::nansum(SnowRain_par.CWmax  * *fAPAR*Mask);	
      } else {
	theta_canopy = torch::nansum(*Interception*Mask + theta_canopy*Mask);
      }     
    }
    std::cout << "aIFcanop";
    std::cout << theta_canopy;
    
    if (LOGFLAG > 1.5) 
      fprintf(flog, "   preles(): estimated canopy water", 
	      theta_canopy);

    std::cout << "bET";
    std::cout << theta_canopy;
    *ET = *ET*mask + ETfun(*VPD, theta, *PAR, *fAPAR, *TAir, 
			   ET_par, Site_par,
			   &theta_canopy,
			   &fE[0], // Soil water constrain on evaporation  
			   gpp380, 
			   *fW, // soil water constrain of GPP at 380 ppm
			   GPP_par, //fCO2_ET_model_mean(CO2[i], GPP_par),
			   *CO2, 
			   flog, LOGFLAG, etmodel, 
			   &transp[0], 
			   &evap[0], &fWE[0], i, Mask, mask);
    std::cout << "aET";
    std::cout << theta_canopy;
    
    if (LOGFLAG > 1.5) 
      fprintf(flog, 
	      "   preles(): ET=%lf\n", ET[i]);

     
      /* Calculate soil water balance, drainage and related variables at the 
         end of the day, as well as update snow and canopy water with et */
    
    //    swbalance(&theta, Throughfall[i], Snowmelt[i], ET[i],
    std::cout << "PRESWB";
    std::cout << theta_snow;
    std::cout << theta_canopy;
    swbalance(&theta, *Throughfall, *Snowmelt, *ET,
	      Site_par, &Drainage[0], //&Psi[i], &Ks[i], 
	      &theta_snow, &theta_canopy, SnowRain_par, i, Mask, mask);
    std::cout << "POSTSWB";
    std::cout << theta_snow;
    std::cout << theta_canopy;
    if (LOGFLAG > 1.5) 
      fprintf(flog, 
	      "   preles(): drainage, after ET: SW \tSOG CW\n", 
	      Drainage[i], theta, theta_snow, theta_canopy);
    
    /* Record result variables with storage components */
    *SOG = *SOG*mask + theta_snow*Mask;
    *SW = *SW*mask + theta*Mask;
    *Canopywater = *Canopywater*mask + theta_canopy*Mask;
    *S = *S*mask + S_state*Mask;

    std::cout << "CO2";
    std::cout << *CO2;
    
    if (LOGFLAG > 1.5) 
      fprintf(flog, 
	      "   preles(): after day state:\n   SW \tCW \tSOG \tS \n\n", SW[i], Canopywater[i], SOG[i], S[i]);

  } // END DAY LOOP
  

  if (LOGFLAG > 1.5) 
    fprintf(flog, 
	    "   preles(): looped all days, closing preles.log, exiting...\n");
  if (flog) fclose(flog);
  
  return(0);
}
    
