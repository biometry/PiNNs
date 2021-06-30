#include "prelesglobals.h"
#include <pybind11/pybind11.h>
#include <TH/TH.h>
#include <stdio.h>

/* Replace missing first day values with something reasoble if missing */
void initConditions(torch::Tensor **PAR, torch::Tensor **TAir, torch::Tensor **VPD, torch::Tensor **Precip, 
                    torch::Tensor **CO2, torch::Tensor Msk, torch::Tensor msk) {
  /* if first day value is missing assume we're somewhere on the boreal
   * zone (lat > 60 deg) */
  if (torch::any(torch::lt(**PAR*Msk, -900)).item<bool>()) **PAR=**PAR*msk + torch::tensor(5., torch::requires_grad()); // it was a dark winter day...
  if (torch::any(torch::lt(**TAir*Msk, -900)).item<bool>()) **TAir=**TAir*msk + torch::tensor(0., torch::requires_grad());
  if (torch::any(torch::lt(**VPD*Msk, 0)).item<bool>() || torch::any(torch::gt(**VPD*Msk, 6)).item<bool>()) **VPD=**VPD*msk + torch::tensor(0.5, torch::requires_grad()); // VPD > 6 implausible, 3 = very dry air
  if (torch::any(torch::lt(**Precip*Msk, 0)).item<bool>()) **Precip=**Precip*msk + torch::tensor(0., torch::requires_grad());
  if (torch::any(torch::lt(**CO2*Msk, 0.1*Msk)).item<bool>()) **CO2=**CO2*msk + torch::tensor(380., torch::requires_grad()); //
  

};
