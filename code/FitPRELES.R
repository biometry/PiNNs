## Fit PRELES to site data

#### install/load relevant packages ####
#devtools::install_github('MikkoPeltoniemi/Rpreles')
library(Rpreles)
library(BayesianTools)

source("helpers.R")

#set flags
make_nas_data = FALSE
ex_fit = FALSE
save_data = FALSE

if (make_nas_data){create_nas_data()}
if (ex_fit){example_fit()}  

#library(mvtnorm)
#dd <- dmvnorm(qq, mean=apply(mm, 2, mean),  log=T)

##========================##
## Singlesite Calibration ##
##========================##

makesparse <- function(train){
  ind <- seq(0,nrow(train), by=7)
  tsmall <- train[ind,]
  return(tsmall)
}


singlesite_calibration <- function(data_use, save_data=FALSE){
  
  load("~/Projects/physics_guided_nn/data/parameterRanges.rdata") # parameter defaults/ranges
  # par # note that "-999" is supposed to indiate NA!
  pars <- par # unfortunate naming "par" replaced by "pars"
  rm(par)
  pars[pars=="-999"] <- NA
  pars # note that some parameters are set without uncertainty (e.g. soildepth)
  
  # jetzt neu: S[max]
  #pars[pars$name=="S[max]", 4] <- 45 # was 30
  #  S[max]: tick
  pars[pars$name=="nu", 4] <- 10 # was 5

  
  # select the parameters to be calibrated:
  pars2tune <- c(5:11, 14:18, 31) # note that we omit 32, as it refers to ET
  thispar <- pars$def
  names(thispar) <- pars$name
  
  
  hyytiala <- read.csv("~/PycharmProjects/physics_guided_nn/data/hyytiala.csv")
  hyytiala$date <- as.Date(hyytiala$date)
  hyytiala$year <- format(hyytiala$date, format="%Y")
  
  hyytiala_train <- hyytiala[!(hyytiala$year %in% c("2008", "2007", "2005", "2004")), ]
  hyytiala_test <- hyytiala[hyytiala$year == "2008", ]
  attach(hyytiala_train)
  
  
  if (data_use == 'sparse'){
    hyytiala_train <- makesparse(hyytiala_train)
  }
  
  CVfit <- matrix(NA, nrow=nrow(pars), ncol = length(unique(hyytiala_train$year)))
  
  i <- 1
  for (year in unique(hyytiala_train$year)){
  
    df <- hyytiala_train[hyytiala_train$year != year,]
    
    ell <- function(pars, data=df){
      # pars is a vector the same length as pars2tune
      thispar[pars2tune] <- pars
      # likelihood function, first shot: normal density
      with(data, sum(dnorm(df$GPP, mean=PRELES(PAR=df$PAR, TAir=df$Tair, VPD=df$VPD, Precip=df$Precip, CO2=df$CO2, fAPAR=df$fapar , p=thispar)$GPP, sd=thispar[31], log=T)))
    }
    priors <- createUniformPrior(lower=pars$min[pars2tune], upper=pars$max[pars2tune], best=pars$def[pars2tune])
    setup <- createBayesianSetup(likelihood=ell, prior=priors, parallel=T)
    settings <- list(iterations=50000, adapt=T, nrChains=3, parallel=T) # runs 3 chains in parallel for each chain ...
    # run:
    fit <- runMCMC(bayesianSetup = setup, settings = settings, sampler = "DREAMzs")
    save(fit, file = paste0("~/PycharmProjects/physics_guided_nn/data/Psinglesite_fit_", year,"_", data_use, ".Rdata"))
    
    pars_fit <- pars
    pars_fit$def[pars2tune] <- MAP(fit)$parametersMAP
    CVfit[,i] <- pars_fit$def
    i = i+1
  }
  
  save(CVfit, file = paste0("~/PycharmProjects/physics_guided_nn/data/Psinglesite_CVfit_", data_use, ".Rdata"))
  write.csv(CVfit, file=paste0("~/PycharmProjects/physics_guided_nn/data/Psinglesite_CVfit_", data_use, ".csv"))
  
   
  gpp_train <- matrix(NA, nrow=nrow(hyytiala_train), ncol=length(unique(hyytiala_train$year)))
  gpp_test <- matrix(NA, nrow=nrow(hyytiala_test), ncol=length(unique(hyytiala_train$year)))
  et_train <- matrix(NA, nrow=nrow(hyytiala_train), ncol=length(unique(hyytiala_train$year)))
  et_test <- matrix(NA, nrow=nrow(hyytiala_test), ncol=length(unique(hyytiala_train$year)))
  sw_train <- matrix(NA, nrow=nrow(hyytiala_train), ncol=length(unique(hyytiala_train$year)))
  sw_test <- matrix(NA, nrow=nrow(hyytiala_test), ncol=length(unique(hyytiala_train$year)))
  
  load(file = paste0("~/PycharmProjects/physics_guided_nn/data/Psinglesite_CVfit_", data_use, ".Rdata"))
  i <- 1
  for (year in unique(hyytiala_train$year)){
    
    load(file = paste0("~/PycharmProjects/physics_guided_nn/data/Psinglesite_fit_", year,"_", data_use, ".Rdata"))
  
    gpp_train[,i] <- PRELES(PAR=hyytiala_train$PAR, TAir=hyytiala_train$Tair, VPD=hyytiala_train$VPD, Precip=hyytiala_train$Precip, CO2=hyytiala_train$CO2, fAPAR=hyytiala_train$fapar, p=CVfit[,i])$GPP
    gpp_test[,i] <- PRELES(PAR=hyytiala_test$PAR, TAir=hyytiala_test$Tair, VPD=hyytiala_test$VPD, Precip=hyytiala_test$Precip, CO2=hyytiala_test$CO2, fAPAR=hyytiala_test$fapar, p=CVfit[,i])$GPP
    
    et_train[,i] <- PRELES(PAR=hyytiala_train$PAR, TAir=hyytiala_train$Tair, VPD=hyytiala_train$VPD, Precip=hyytiala_train$Precip, CO2=hyytiala_train$CO2, fAPAR=hyytiala_train$fapar, p=CVfit[,i])$ET
    et_test[,i] <- PRELES(PAR=hyytiala_test$PAR, TAir=hyytiala_test$Tair, VPD=hyytiala_test$VPD, Precip=hyytiala_test$Precip, CO2=hyytiala_test$CO2, fAPAR=hyytiala_test$fapar, p=CVfit[,i])$ET
    
    sw_train[,i] <- PRELES(PAR=hyytiala_train$PAR, TAir=hyytiala_train$Tair, VPD=hyytiala_train$VPD, Precip=hyytiala_train$Precip, CO2=hyytiala_train$CO2, fAPAR=hyytiala_train$fapar, p=CVfit[,i])$SW
    sw_test[,i] <- PRELES(PAR=hyytiala_test$PAR, TAir=hyytiala_test$Tair, VPD=hyytiala_test$VPD, Precip=hyytiala_test$Precip, CO2=hyytiala_test$CO2, fAPAR=hyytiala_test$fapar, p=CVfit[,i])$SW
    
    i <- i+1
  }
  
  ## Update data set with new calibrated Preles predictions ##
  
  hyytiala_train$GPPp <- apply(gpp_train, 1, mean)
  hyytiala_test$GPPp <- apply(gpp_test, 1, mean)
  hyytiala_train$ETp <- apply(et_train, 1, mean)
  hyytiala_test$ETp <- apply(et_test, 1, mean)
  hyytiala_train$SWp <- apply(sw_train, 1, mean)
  hyytiala_test$SWp <- apply(sw_test, 1, mean)
  
  if (save_data){
    if (data_use == 'full'){
      hyytialaF <- rbind(hyytiala_train, hyytiala_test)
      write.csv(hyytialaF, file="~/PycharmProjects/physics_guided_nn/data/hyytialaF.csv", row.names = FALSE)
      ## Generate files for prediction results ##
      save(gpp_train, file = "~/PycharmProjects/physics_guided_nn/data/GPPp_singlesite_train.Rdata")
      save(gpp_test, file = "~/PycharmProjects/physics_guided_nn/data/GPPp_singlesite_test.Rdata")
    }
  }
  
  
  GPP_train <- apply(gpp_train, 1, mean)
  GPP_test <- apply(gpp_test, 1, mean)
  GPP_train_std <- apply(gpp_train, 1, sd)
  GPP_test_std <- apply(gpp_test, 1, sd)
  
  mae <- function(yhat){
    mae <- sum(abs(hyytiala_test$GPP - yhat))/length(yhat)
    return(mae)
  }
  rmse <- function(yhat){
    rmse <- sqrt(sum((hyytiala_test$GPP - yhat)^2)/length(yhat))
    return(rmse)
  }
  
  perfpormance_preles_full <- matrix(NA, nrow=4, ncol=2)
  perfpormance_preles_full[,1] <- apply(gpp_test, 2, rmse)
  perfpormance_preles_full[,2] <- apply(gpp_test, 2, mae)
  
  write.csv(perfpormance_preles_full, file=paste0("~/PycharmProjects/physics_guided_nn/results_final/preles_eval_", data_use, "_performance.csv"))
  write.csv(gpp_test, file=paste0("~/PycharmProjects/physics_guided_nn/results_final/preles_eval_preds_test_", data_use, ".csv"))
  
}

#load("~/Projects/physics_guided_nn/data/parameterRanges.rdata") # parameter defaults/ranges
#pars2tune <- c(5:11, 14:18, 31) 
#CVfit <- matrix(NA, nrow=nrow(par),ncol = 4)
#i = 1
#for (year in c(2009, 2010, 2011, 2012)){
#  load(file = paste0("~/PycharmProjects/physics_guided_nn/data/Psinglesite_fit_", year,"_", data_use, ".Rdata"))
#  CVfit[,i] <- par$def
#  CVfit[pars2tune,i] <- MAP(fit)$parametersMAP
#  i = i+1
#}

#save(CVfit, file = paste0("~/PycharmProjects/physics_guided_nn/data/Psinglesite_CVfit_", data_use, ".Rdata"))
#write.csv(CVfit, file=paste0("~/PycharmProjects/physics_guided_nn/data/Psinglesite_CVfit_", data_use, ".csv"))

singlesite_calibration(data_use = 'full')
singlesite_calibration(data_use = 'sparse')

##=======================##
## Multisite Calibration ##
##=======================##


multisite_calibration <- function(data_use, save_data = FALSE){
  
  #load("EddyCovarianceDataBorealSites.rdata") # data for one site: s1-s4
  #attach(s1)
  allsites <- read.csv("~/PycharmProjects/physics_guided_nn/data/allsites.csv")
  allsites$date <- as.Date(allsites$date)
  allsites$year <- format(allsites$date, format="%Y")
  print(unique(allsites$year))
  allsites$site <- substr(allsites$X, 1, 1)
  
  allsites_train <- allsites[(allsites$year %in% c("2005", "2004")), ]
  allsites_test <- allsites[allsites$year == "2008", ]
  attach(allsites_train)
  
  summary(allsites_train)
  
  
  if (data_use == 'sparse'){
    allsites_train <- makesparse(allsites_train)
  }
  
  
  load("~/PycharmProjects/physics_guided_nn/data/parameterRanges.rdata") # parameter defaults/ranges
  # par # note that "-999" is supposed to indiate NA!
  pars <- par # unfortunate naming "par" replaced by "pars"
  rm(par)
  pars[pars=="-999"] <- NA
  pars # note that some parameters are set without uncertainty (e.g. soildepth)
  
  pars[pars$name=="nu", 4] <- 10 # was 5
  pars2tune <- c(5:11, 14:18, 31) # note that we omit 32, as it refers to ET
  thispar <- pars$def
  names(thispar) <- pars$name
  
  #### Bayesian Fitting ####

  CVfit <- matrix(NA, nrow=nrow(pars), ncol = length(unique(allsites_train$site)))
  
  i <- 1
  for (s in unique(allsites_train$site)){
    
    df <- allsites_train[allsites_train$site != s,]
    print(s)
    
    ell <- function(pars, data=df){
      # pars is a vector the same length as pars2tune
      thispar[pars2tune] <- pars
      # likelihood function, first shot: normal density
      with(data, sum(dnorm(df$GPP, mean=PRELES(PAR=df$PAR, TAir=df$Tair, VPD=df$VPD, Precip=df$Precip, CO2=df$CO2, fAPAR=df$fapar , p=thispar)$GPP, sd=thispar[31], log=T)))
    }
    priors <- createUniformPrior(lower=pars$min[pars2tune], upper=pars$max[pars2tune], best=pars$def[pars2tune])
    setup <- createBayesianSetup(likelihood=ell, prior=priors, parallel=T)
    settings <- list(iterations=50000, adapt=T, nrChains=3, parallel=T) # runs 3 chains in parallel for each chain ...
    # run:
    fit <- runMCMC(bayesianSetup = setup, settings = settings, sampler = "DREAMzs")
    save(fit, file = paste0("~/PycharmProjects/physics_guided_nn/data/Pmultisite_fit_", s, "_", data_use, ".Rdata"))
    
    pars_fit <- pars
    pars_fit$def[pars2tune] <- MAP(fit)$parametersMAP
    CVfit[,i] <- pars_fit$def
    
    i = i+1
  }
  
  save(CVfit, file = paste0("~/PycharmProjects/physics_guided_nn/data/Pmultisite_CVfit_", data_use, ".Rdata"))
  write.csv(CVfit, file=paste0("~/PycharmProjects/physics_guided_nn/data/Pmultisite_CVfit_", data_use, ".csv"))
  
  testlength <- nrow(allsites[allsites_test$site==unique(allsites_test$site)[1],])
  
  gpp_train <- matrix(NA, nrow=nrow(allsites_train), ncol=length(unique(allsites_train$site)))
  gpp_test <- matrix(NA, nrow=testlength, ncol=length(unique(allsites_train$site)))
  et_train <- matrix(NA, nrow=nrow(allsites_train), ncol=length(unique(allsites_train$site)))
  et_test <- matrix(NA, nrow=testlength, ncol=length(unique(allsites_train$site)))
  sw_train <- matrix(NA, nrow=nrow(allsites_train), ncol=length(unique(allsites_train$site)))
  sw_test <- matrix(NA, nrow=testlength, ncol=length(unique(allsites_train$site)))
  
  load(file = paste0("~/PycharmProjects/physics_guided_nn/data/Pmultisite_CVfit_", data_use, ".Rdata"))
  i <- 1
  for (s in unique(allsites_train$site)){
    
    load(file = paste0("~/PycharmProjects/physics_guided_nn/data/Pmultisite_fit_", s, "_", data_use, ".Rdata"))
    
    test_df <- allsites[allsites_test$site==s,]
    
    gpp_train[,i] <- PRELES(PAR=allsites_train$PAR, TAir=allsites_train$Tair, VPD=allsites_train$VPD, Precip=allsites_train$Precip, CO2=allsites_train$CO2, fAPAR=allsites_train$fapar, p=CVfit[,i])$GPP
    gpp_test[,i] <- PRELES(PAR=test_df$PAR, TAir=test_df$Tair, VPD=test_df$VPD, Precip=test_df$Precip, CO2=test_df$CO2, fAPAR=test_df$fapar, p=CVfit[,i])$GPP
    
    et_train[,i] <- PRELES(PAR=allsites_train$PAR, TAir=allsites_train$Tair, VPD=allsites_train$VPD, Precip=allsites_train$Precip, CO2=allsites_train$CO2, fAPAR=allsites_train$fapar, p=CVfit[,i])$ET
    et_test[,i] <- PRELES(PAR=test_df$PAR, TAir=test_df$Tair, VPD=test_df$VPD, Precip=test_df$Precip, CO2=test_df$CO2, fAPAR=test_df$fapar, p=CVfit[,i])$ET
    
    sw_train[,i] <- PRELES(PAR=allsites_train$PAR, TAir=allsites_train$Tair, VPD=allsites_train$VPD, Precip=allsites_train$Precip, CO2=allsites_train$CO2, fAPAR=allsites_train$fapar, p=CVfit[,i])$SW
    sw_test[,i] <- PRELES(PAR=test_df$PAR, TAir=test_df$Tair, VPD=test_df$VPD, Precip=test_df$Precip, CO2=test_df$CO2, fAPAR=test_df$fapar, p=CVfit[,i])$SW
    
    i <- i+1
  }
  
  ## Update data set with new calibrated Preles predictions ##
  
  allsites_train$GPPp <- apply(gpp_train, 1, mean)
  allsites_test$GPPp <- apply(gpp_test, 1, mean)
  allsites_train$ETp <- apply(et_train, 1, mean)
  allsites_test$ETp <- apply(et_test, 1, mean)
  allsites_train$SWp <- apply(sw_train, 1, mean)
  allsites_test$SWp <- apply(sw_test, 1, mean)
  
  i <-1
  for (site in unique(allsites_test$site)){
     allsites[allsites_test$site==site,]$GPPp <- gpp_test[,i]
     allsites[allsites_test$site==site,]$ETp <- et_test[,i]
     allsites[allsites_test$site==site,]$SWp <- sw_test[,i]
     i <- i+1
  }
  
  if (save_data){
    allsitesF <- rbind(allsites_train, allsites_test)
    
    if (data_use== 'full'){
      write.csv(allsitesF, file="~/PycharmProjects/physics_guided_nn/data/allsitesF.csv", row.names = FALSE)
    }
    
    pdf(file="~/PycharmProjects/physics_guided_nn/results/Pmultisitefit_BayesPriors.pdf", width=15, height=12)
    par(mfrow=c(4, 4), mar=c(3,3,3,1))
    for (i in 1:ncol(fit[[1]]$X)){ # loop over parameters fitted
      #fit1[[1]]$chain[1]
      # for DEzs:
      ests <- rbind(fit[[1]]$Z, fit[[2]]$Z, fit[[3]]$Z)
      plot(density(ests[,i], from=min(ests[,i]), to=max(ests[,i])), main=pars[pars2tune[i],1], las=1)
      abline(v=pars[pars2tune[i], 3:4], col="red")
    }
    dev.off()
  }
  
  
  
  ## Generate files for prediction results ##
  
  save(gpp_train, file = paste0("~/PycharmProjects/physics_guided_nn/data/2GPPp_train_", data_use, ".Rdata"))
  save(gpp_test, file = paste0("~/PycharmProjects/physics_guided_nn/data/2GPPp_test_", data_use, ".Rdata"))
  
  
  GPP_train <- apply(gpp_train, 1, mean)
  GPP_train_std <- apply(gpp_train, 1, sd)
  
  mae <- function(yhat, test=T, site = NA){
    if (test){
      mae <- sum(abs(allsites_test[allsites_test$site == site,]$GPP - yhat))/length(yhat)
    }else{
      mae <- sum(abs(allsites_train$GPP - yhat))/length(yhat)
    }
    return(mae)
  }
  rmse <- function(yhat, test=T, site=NA){
    if (test){
      rmse <- sqrt(sum((allsites_test[allsites_test$site == site,]$GPP - yhat)^2)/length(yhat))
    }else{
      rmse <- sqrt(sum((allsites_train$GPP - yhat)^2)/length(yhat))
    }
    return(rmse)
  }
  
  perfpormance_preles_full <- matrix(NA, nrow=length(unique(allsites_train$site)), ncol=4)
  perfpormance_preles_full[,1] <- apply(gpp_train, 2, rmse, test=F)
  i <-1
  for (site in unique(allsites_test$site)){
    perfpormance_preles_full[i,2] <- apply(gpp_test[,i], 2, rmse, test=T, site=site)
    i <- i+1
  }
  perfpormance_preles_full[,3] <- apply(gpp_train, 2, mae, test=F)
  i <-1
  for (site in unique(allsites_test$site)){
    perfpormance_preles_full[i,2] <- apply(gpp_test[,i], 2, rmse, test=T, site=site)
    i <- i+1
  }
  
  write.csv(perfpormance_preles_full, file=paste0("~/PycharmProjects/physics_guided_nn/results_final/2preles_eval_", data_use, "_performance.csv"))
  write.csv(gpp_test, file=paste0("~/PycharmProjects/physics_guided_nn/results_final/2preles_eval_preds_test_", data_use, "2.csv"))
}

multisite_calibration(data_use = 'sparse')
multisite_calibration(data_use = 'full')
