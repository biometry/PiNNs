
#ifndef PRELESGLOBALS_H
#define	PRELESGLOBALS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <torch/extension.h>

#define TARGACCEPT 0.44
#define MAXK 1000
#define MYINFINITY 999999999.9
#define PI 3.1415926535
#define NUMBER_OF_MODEL_PARAMETERS 38

//int K;
//int vectorlength;

/* Site soil and other parameters, some in input, some calculated in code */
typedef struct p1 {  
  torch::Tensor soildepth; 
  torch::Tensor ThetaFC;  
  torch::Tensor ThetaPWP;  
  torch::Tensor tauDrainage;
} p1 ;

//  GPP model 
typedef struct p2 {
  torch::Tensor beta; 
  torch::Tensor tau; 
  torch::Tensor S0;
  torch::Tensor Smax;
  torch::Tensor kappa;
  torch::Tensor gamma;
  torch::Tensor soilthres; // used for fW with ETmodel = 2 | 4 | 6
  torch::Tensor bCO2; // used for fW with ETmodel = 1 | 3 | 5
  torch::Tensor xCO2; // used for fW with ETmodel = 1 | 3 | 5;
  torch::Tensor t0; // Birch phenology parameters: 26th Feb = 57 DOY
  torch::Tensor tcrit; // (Linkosalo et al)           1.5 C
  torch::Tensor tsumcrit; //                                 134  C
  
} p2 ;

// ET-model
typedef struct p3 {
  torch::Tensor beta; 
  torch::Tensor kappa; 
  torch::Tensor chi;
  torch::Tensor soilthres; // used for fW with ETmodel = 2 | 4
  torch::Tensor nu; 
} p3 ;

// Rain and Snow models: interception and melting of snow 
typedef struct p4 {
  torch::Tensor MeltCoef; 
 // torch::Tensor Ifrac;
  torch::Tensor I0; 
  torch::Tensor CWmax;
  torch::Tensor SnowThreshold;
  torch::Tensor T_0;
} p4; 

// Storage components
typedef struct p5 {
  torch::Tensor SW; // Soilw water at beginning
  torch::Tensor CW; // Canopy water
  torch::Tensor SOG; // Snow on Ground 
  torch::Tensor S; // State of temperature acclimation
} p5; 

typedef struct p6 {
  torch::Tensor cvGPP; // Coefficients of variation for GPP, ET and SW
  torch::Tensor cvET;  // Used in MCMC-calibration only
  torch::Tensor cvSW; 
} p6; 


#ifdef	__cplusplus
extern "C" {
#endif


#ifdef	__cplusplus
}
#endif

#endif	/* PRELESGLOBALS_H */

