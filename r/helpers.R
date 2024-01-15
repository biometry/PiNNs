library(BayesianTools)
library(Rpreles)

create_nas_data <- function(pars_def = TRUE, write = FALSE){
  
  ##=========================##
  ## Create data set for NAS ##
  ##=========================##
  
  hyytiala <- read.csv("~/PycharmProjects/physics_guided_nn/data/hyytiala.csv")
  hyytiala$date <- as.Date(hyytiala$date)
  hyytiala$year <- format(hyytiala$date, format="%Y")
  hyytiala_nas <- hyytiala[(hyytiala$year %in% c( "2005", "2004")), ]
  attach(hyytiala_nas)
  
  load("~/PycharmProjects/physics_guided_nn/data/parameterRanges.rdata") # parameter defaults/ranges
  # par # note that "-999" is supposed to indiate NA!
  pars <- par # unfortunate naming "par" replaced by "pars"
  rm(par)
  pars[pars=="-999"] <- NA
  pars # note that some parameters are set without uncertainty (e.g. soildepth)
  pars[pars$name=="nu", 4] <- 10 # was 5
  
  if (pars_def != TRUE){
  
    #### Bayesian Fitting ####
    # select the parameters to be calibrated:
    pars2tune <- c(5:11, 14:18, 31) # note that we omit 32, as it refers to ET
    thispar <- pars$def
    names(thispar) <- pars$name
    
    gpp <-  PRELES(PAR=PAR, TAir=Tair, VPD=VPD, Precip=Precip, CO2=CO2, fAPAR=fapar, p=thispar)$GPP #, 
    et <- PRELES(PAR=PAR, TAir=Tair, VPD=VPD, Precip=Precip, CO2=CO2, fAPAR=fapar, p=thispar)$ET
    #sw <- PRELES(PAR=PAR, TAir=Tair, VPD=VPD, Precip=Precip, CO2=CO2, fAPAR=fapar, p=thispar)$SW
    #qq <- cbind(GPP, ET)
    
    #library(mvtnorm)
    #dd <- dmvnorm(qq, mean=apply(mm, 2, mean),  log=T)
    
    ell <- function(pars, data=hyytiala_nas){
      # pars is a vector the same length as pars2tune
      thispar[pars2tune] <- pars
      # likelihood function, first shot: normal density
      with(data, (sum(dnorm(GPP, mean=gpp, sd = thispar[31], log=T)) + 
                    sum(dnorm(ET, mean=et, sd = thispar[31], log=T))))
      #with(data, (sum(dnorm(GPP, mean=gpp, sd = thispar[31], log=T))))
    }
    priors <- createUniformPrior(lower=pars$min[pars2tune], upper=pars$max[pars2tune], best=pars$def[pars2tune])
    setup <- createBayesianSetup(likelihood=ell, prior=priors, parallel=T)
    settings <- list(iterations=50000, adapt=T, nrChains=3, parallel=T) # runs 3 chains in parallel for each chain ...
    # run:
    fit <- runMCMC(bayesianSetup = setup, settings = settings, sampler = "DREAMzs")
    save(fit, file = "~/PycharmProjects/physics_guided_nn/data/Psinglesite_NAS_fit.Rdata")
    summary(fit)
    
    pars_fit <- pars
    pars_fit$def[pars2tune] <- MAP(fit)$parametersMAP
    
    ## Add fitted parameter values:
    hyytiala_nas$GPPp <- PRELES(PAR=hyytiala_nas$PAR, TAir=hyytiala_nas$Tair, VPD=hyytiala_nas$VPD, Precip=hyytiala_nas$Precip, CO2=hyytiala_nas$CO2, fAPAR=hyytiala_nas$fapar, p=pars_fit$def)$GPP
    hyytiala_nas$ETp <- PRELES(PAR=hyytiala_nas$PAR, TAir=hyytiala_nas$Tair, VPD=hyytiala_nas$VPD, Precip=hyytiala_nas$Precip, CO2=hyytiala_nas$CO2, fAPAR=hyytiala_nas$fapar, p=pars_fit$def)$ET
    hyytiala_nas$SWp <- PRELES(PAR=hyytiala_nas$PAR, TAir=hyytiala_nas$Tair, VPD=hyytiala_nas$VPD, Precip=hyytiala_nas$Precip, CO2=hyytiala_nas$CO2, fAPAR=hyytiala_nas$fapar, p=pars_fit$def)$SW
    ## Very bad fit!
    mae <- sum(abs(hyytiala_nas$GPP - hyytiala_nas$GPPp))/length(hyytiala_nas$GPPp)
    plot(hyytiala_nas$GPPp)
    
  }else{
      
    ## Simply use default parameter values.
    hyytiala_nas$GPPp <- PRELES(PAR=hyytiala_nas$PAR, TAir=hyytiala_nas$Tair, VPD=hyytiala_nas$VPD, Precip=hyytiala_nas$Precip, CO2=hyytiala_nas$CO2, fAPAR=hyytiala_nas$fapar, p=pars$def)$GPP
    hyytiala_nas$ETp <- PRELES(PAR=hyytiala_nas$PAR, TAir=hyytiala_nas$Tair, VPD=hyytiala_nas$VPD, Precip=hyytiala_nas$Precip, CO2=hyytiala_nas$CO2, fAPAR=hyytiala_nas$fapar, p=pars$def)$ET
    hyytiala_nas$SWp <- PRELES(PAR=hyytiala_nas$PAR, TAir=hyytiala_nas$Tair, VPD=hyytiala_nas$VPD, Precip=hyytiala_nas$Precip, CO2=hyytiala_nas$CO2, fAPAR=hyytiala_nas$fapar, p=pars$def)$SW
    ## Much better fit.
    mae <- sum(abs(hyytiala_nas$GPP - hyytiala_nas$GPPp))/length(hyytiala_nas$GPPp)
    plot(hyytiala_nas$GPPp)
    
  }
  
  if (write){
    write.csv(hyytiala_nas, file="~/PycharmProjects/physics_guided_nn/data/hyytialaNAS.csv", row.names = FALSE)
  }
  
  return(hyytiala_nas, mae)
    
}

example_fit <- function(){
  
  hyytiala <- read.csv("~/PycharmProjects/physics_guided_nn/data/hyytiala.csv")
  hyytiala$date <- as.Date(hyytiala$date)
  hyytiala$year <- format(hyytiala$date, format="%Y")
  
  hyytiala_train <- hyytiala[!(hyytiala$year %in% c("2008", "2007", "2005", "2004")), ]
  hyytiala_test <- hyytiala[hyytiala$year == "2008", ]
  attach(hyytiala_train)
  
  load("~/Projects/physics_guided_nn/data/parameterRanges.rdata") # parameter defaults/ranges
  # par # note that "-999" is supposed to indiate NA!
  pars <- par # unfortunate naming "par" replaced by "pars"
  rm(par)
  pars[pars=="-999"] <- NA
  pars # note that some parameters are set without uncertainty (e.g. soildepth)
  
  # add min/max for some of the parameters so far missing?
  
  # RESULTS FROM CARSTENS FIRST RUN
  # these findings differ significantly for different data.
  
  # potentially problematic:
  # tau, gamma, rho[P], alpha, lambda, chi, nu
  #pars[pars$name=="tau", 4] <- 35 # was 25
  #pars[pars$name=="gamma", 3] <- 1e-5 # was 0.000103
  # pars[pars$name=="rho[P]", 4] <- 0.999, limit is probably 1!!
  # pars[pars$name=="alpha", ] is already 1E-6
  #pars[pars$name=="lambda", 4] <- 2 # was 1.2
  # pars[pars$name=="chi", ] is already down to 0
  #pars[pars$name=="nu", 4] <- 7.5 # was 5
  
  # re-running this does not work: 
  #  tau: tick
  #  gamma: tick
  #  rho: nee, Obergrenze ist so schon maximal
  #  lambda: tick
  #  nu: tick
  
  # jetzt neu: S[max]
  #pars[pars$name=="S[max]", 4] <- 45 # was 30
  #  S[max]: tick
  pars[pars$name=="nu", 4] <- 10 # was 5
  
  #### EXAMPLE RUN ####
  #onerun <- PRELES(PAR=PAR, TAir=Tair, VPD=VPD, Precip=Precip, CO2=CO2, fAPAR=fapar, p=pars[,"def"])
  # make a plot of the output:
  #par(mfrow=c(3,1), mar=c(2,4,1,1), oma=c(4,0,0,0))
  #plot(1:(2*365), onerun$GPP, type="l", las=1, ylab="GPP")
  #abline(v=366)
  #plot(1:(2*365), onerun$ET, type="l", las=1, ylab="evapotranspiration")
  #abline(v=366)
  #plot(1:(2*365), onerun$SW, type="l", las=1, ylab="soil water")
  #abline(v=366)
  #mtext(side=1, line=4, "day since start")
  
  #### Bayesian Fitting ####
  
  # select the parameters to be calibrated:
  pars2tune <- c(5:11, 14:18, 31) # note that we omit 32, as it refers to ET
  thispar <- pars$def
  names(thispar) <- pars$name
  
  gpp <-  PRELES(PAR=PAR, TAir=Tair, VPD=VPD, Precip=Precip, CO2=CO2, fAPAR=fapar, p=thispar)$GPP #, 
  et <- PRELES(PAR=PAR, TAir=Tair, VPD=VPD, Precip=Precip, CO2=CO2, fAPAR=fapar, p=thispar)$ET
  #qq <- cbind(GPP, ET)
  
  #library(mvtnorm)
  #dd <- dmvnorm(qq, mean=apply(mm, 2, mean),  log=T)
  
  ell <- function(pars, data=hyytiala_train){
    # pars is a vector the same length as pars2tune
    thispar[pars2tune] <- pars
    # likelihood function, first shot: normal density
    #with(data, (sum(dnorm(GPP, mean=gpp, sd = thispar[31], log=T)) + 
    #              sum(dnorm(ET, mean=et, sd = thispar[31], log=T))))
    with(data, (sum(dnorm(GPP, mean=gpp, sd = thispar[31], log=T))))
  }
  priors <- createUniformPrior(lower=pars$min[pars2tune], upper=pars$max[pars2tune], best=pars$def[pars2tune])
  setup <- createBayesianSetup(likelihood=ell, prior=priors, parallel=T)
  settings <- list(iterations=50000, adapt=T, nrChains=3, parallel=T) # runs 3 chains in parallel for each chain ...
  # run:
  fit1 <- runMCMC(bayesianSetup = setup, settings = settings, sampler = "DREAMzs")
  save(fit1, file = "~/PycharmProjects/physics_guided_nn/data/Psinglesite_example_fit.Rdata")
  
  #### Check whether estimates reach prior boundary ####
  summary(fit1)
  pdf(file="~/PycharmProjects/physics_guided_nn/results_final/Psinglesitefit_BayesPriors.pdf", width=15, height=12)
  par(mfrow=c(4, 4), mar=c(3,3,3,1))
  for (i in 1:ncol(fit1[[1]]$X)){ # loop over parameters fitted
    #fit1[[1]]$chain[1]
    # for DEzs:
    ests <- rbind(fit1[[1]]$Z, fit1[[2]]$Z, fit1[[3]]$Z)
    plot(density(ests[,i], from=min(ests[,i]), to=max(ests[,i])), main=pars[pars2tune[i],1], las=1)
    abline(v=pars[pars2tune[i], 3:4], col="red")
  }
  dev.off()
  
  
  ### Predictions to test year 2008 with MAP
  detach(hyytiala_train)
  attach(hyytiala_test)
  
  summary(fit1)
  
  pars_fit <- pars
  pars_fit$def[pars2tune] <- MAP(fit1)$parametersMAP
  p_preds <- PRELES(PAR=PAR, TAir=Tair, VPD=VPD, Precip=Precip, CO2=CO2, fAPAR=fapar, p=pars_fit$def)$GPP
  
  sum(abs(GPP - p_preds))/length(p_preds)
  plot(p_preds, type="l")
  detach(hyytiala_test)
  
}