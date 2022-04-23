## Fit PRELES to site data

#### install/load relevant packages ####
#devtools::install_github('MikkoPeltoniemi/Rpreles')
library(Rpreles)

#load("EddyCovarianceDataBorealSites.rdata") # data for one site: s1-s4
#attach(s1)
hyytiala <- read.csv("~/physics_guided_nn/data/hyytiala.csv")
hyytiala$date <- as.Date(hyytiala$date)
hyytiala$year <- format(hyytiala$date, format="%Y")
hyytiala_train <- hyytiala[!(hyytiala$year %in% c("2008", "2007", "2005", "2004")), ]
hyytiala_test <- hyytiala[hyytiala$year == "2008", ]
attach(hyytiala_train)

load("~/physics_guided_nn/data/parameterRanges.rdata") # parameter defaults/ranges
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
onerun <- PRELES(PAR=PAR, TAir=Tair, VPD=VPD, Precip=Precip, CO2=CO2, fAPAR=fapar, p=pars[,"def"])
# make a plot of the output:
par(mfrow=c(3,1), mar=c(2,4,1,1), oma=c(4,0,0,0))
plot(1:(2*365), onerun$GPP, type="l", las=1, ylab="GPP")
abline(v=366)
plot(1:(2*365), onerun$ET, type="l", las=1, ylab="evapotranspiration")
abline(v=366)
plot(1:(2*365), onerun$SW, type="l", las=1, ylab="soil water")
abline(v=366)
mtext(side=1, line=4, "day since start")

#### Bayesian Fitting ####

library(BayesianTools)
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
  # with(data, (sum(dnorm(GPP, mean=gpp, sd = thispar[31], log=T)) + 
  #              sum(dnorm(ET, mean=et, sd = thispar[31], log=T))))
  with(data, (sum(dnorm(GPP, mean=gpp, sd = thispar[31], log=T))))
}
priors <- createUniformPrior(lower=pars$min[pars2tune], upper=pars$max[pars2tune], best=pars$def[pars2tune])
setup <- createBayesianSetup(likelihood=ell, prior=priors, parallel=T)
settings <- list(iterations=50000, adapt=T, nrChains=3, parallel=T) # runs 3 chains in parallel for each chain ...
# run:
fit1 <- runMCMC(bayesianSetup = setup, settings = settings, sampler = "DREAMzs")
save(fit1, file = "~/physics_guided_nn/data/Psinglesite_fit.Rdata")

#### Check whether estimates reach prior boundary ####
summary(fit1)
pdf(file="~/physics_guided_nn/results/Psinglesitefit_BayesPriors.pdf", width=15, height=12)
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

## Calibrate Preles in CV setting ##
####################################


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
  save(fit, file = paste0("~/physics_guided_nn/data/Psinglesite_fit_", year, ".Rdata"))
  
  pars_fit <- pars
  pars_fit$def[pars2tune] <- MAP(fit)$parametersMAP
  CVfit[,i] <- pars_fit$def
}

save(CVfit, file = "~/physics_guided_nn/data/Psinglesite_CVfit.Rdata")
 
gpp_train <- matrix(NA, nrow=nrow(hyytiala_train), ncol=length(unique(hyytiala_train$year)))
gpp_test <- matrix(NA, nrow=nrow(hyytiala_test), ncol=length(unique(hyytiala_train$year)))
et_train <- matrix(NA, nrow=nrow(hyytiala_train), ncol=length(unique(hyytiala_train$year)))
et_test <- matrix(NA, nrow=nrow(hyytiala_test), ncol=length(unique(hyytiala_train$year)))
sw_train <- matrix(NA, nrow=nrow(hyytiala_train), ncol=length(unique(hyytiala_train$year)))
sw_test <- matrix(NA, nrow=nrow(hyytiala_test), ncol=length(unique(hyytiala_train$year)))

load(file = "~/physics_guided_nn/data/Psinglesite_CVfit.Rdata")
i <- 1
for (year in unique(hyytiala_train$year)){
  
  load(file = paste0("~/physics_guided_nn/data/Psinglesite_fit_", year, ".Rdata"))

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

hyytialaF <- rbind(hyytiala_train, hyytiala_test)

write.csv(hyytialaF, file="~/physics_guided_nn/data/hyytialaF.csv", row.names = FALSE)

## Generate files for prediction results ##

save(gpp_train, file = "~/physics_guided_nn/data/GPPp_singlesite_train.Rdata")
save(gpp_test, file = "~/physics_guided_nn/data/GPPp_singlesite_test.Rdata")


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

write.csv(perfpormance_preles_full, file="~/physics_guided_nn/results/performance_preles_full.csv")
write.csv(gpp_test, file="~/physics_guided_nn/results/preles_eval_preds_test_full.csv")

##=========================##
## Create data set for NAS ##
##=========================##

hyytiala <- read.csv("~/physics_guided_nn/data/hyytiala.csv")
hyytiala$date <- as.Date(hyytiala$date)
hyytiala$year <- format(hyytiala$date, format="%Y")
hyytiala_nas <- hyytiala[(hyytiala$year %in% c( "2005", "2004")), ]
attach(hyytiala_nas)

load("~/physics_guided_nn/data/parameterRanges.rdata") # parameter defaults/ranges
# par # note that "-999" is supposed to indiate NA!
pars <- par # unfortunate naming "par" replaced by "pars"
rm(par)
pars[pars=="-999"] <- NA
pars # note that some parameters are set without uncertainty (e.g. soildepth)
pars[pars$name=="nu", 4] <- 10 # was 5


#### Bayesian Fitting ####

library(BayesianTools)
# select the parameters to be calibrated:
pars2tune <- c(5:11, 14:18, 31) # note that we omit 32, as it refers to ET
thispar <- pars$def
names(thispar) <- pars$name

gpp <-  PRELES(PAR=PAR, TAir=Tair, VPD=VPD, Precip=Precip, CO2=CO2, fAPAR=fapar, p=thispar)$GPP #, 
et <- PRELES(PAR=PAR, TAir=Tair, VPD=VPD, Precip=Precip, CO2=CO2, fAPAR=fapar, p=thispar)$ET
#qq <- cbind(GPP, ET)

#library(mvtnorm)
#dd <- dmvnorm(qq, mean=apply(mm, 2, mean),  log=T)

ell <- function(pars, data=hyytiala_nas){
  # pars is a vector the same length as pars2tune
  thispar[pars2tune] <- pars
  # likelihood function, first shot: normal density
  # with(data, (sum(dnorm(GPP, mean=gpp, sd = thispar[31], log=T)) + 
  #              sum(dnorm(ET, mean=et, sd = thispar[31], log=T))))
  with(data, (sum(dnorm(GPP, mean=gpp, sd = thispar[31], log=T))))
}
priors <- createUniformPrior(lower=pars$min[pars2tune], upper=pars$max[pars2tune], best=pars$def[pars2tune])
setup <- createBayesianSetup(likelihood=ell, prior=priors, parallel=T)
settings <- list(iterations=50000, adapt=T, nrChains=3, parallel=T) # runs 3 chains in parallel for each chain ...
# run:
fit <- runMCMC(bayesianSetup = setup, settings = settings, sampler = "DREAMzs")
save(fit, file = "~/physics_guided_nn/data/Psinglesite_NAS_fit.Rdata")
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

## Simply use default parameter values.
hyytiala_nas$GPPp <- PRELES(PAR=hyytiala_nas$PAR, TAir=hyytiala_nas$Tair, VPD=hyytiala_nas$VPD, Precip=hyytiala_nas$Precip, CO2=hyytiala_nas$CO2, fAPAR=hyytiala_nas$fapar, p=pars$def)$GPP
hyytiala_nas$ETp <- PRELES(PAR=hyytiala_nas$PAR, TAir=hyytiala_nas$Tair, VPD=hyytiala_nas$VPD, Precip=hyytiala_nas$Precip, CO2=hyytiala_nas$CO2, fAPAR=hyytiala_nas$fapar, p=pars$def)$ET
hyytiala_nas$SWp <- PRELES(PAR=hyytiala_nas$PAR, TAir=hyytiala_nas$Tair, VPD=hyytiala_nas$VPD, Precip=hyytiala_nas$Precip, CO2=hyytiala_nas$CO2, fAPAR=hyytiala_nas$fapar, p=pars$def)$SW
## Much better fit.
mae <- sum(abs(hyytiala_nas$GPP - hyytiala_nas$GPPp))/length(hyytiala_nas$GPPp)
plot(hyytiala_nas$GPPp)

write.csv(hyytiala_nas, file="~/physics_guided_nn/data/hyytialaNAS.csv", row.names = FALSE)

##=======================##
## Multisite Calibration ##
##=======================##


#load("EddyCovarianceDataBorealSites.rdata") # data for one site: s1-s4
#attach(s1)
allsites <- read.csv("~/physics_guided_nn/data/data_exp2.csv")
allsites$date <- as.Date(allsites$date)
allsites$year <- format(allsites$date, format="%Y")
print(unique(allsites$year))
allsites$site <- substr(allsites$X, 1, 1)

allsites_train <- allsites[(allsites$year %in% c("2005", "2004")), ]
allsites_test <- allsites[allsites$year == "2008", ]
attach(allsites_train)

summary(allsites_train)


load("~/physics_guided_nn/data/parameterRanges.rdata") # parameter defaults/ranges
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
library(BayesianTools)

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
  save(fit, file = paste0("~/physics_guided_nn/data/Pmultisite_fit_", s, ".Rdata"))
  
  pars_fit <- pars
  pars_fit$def[pars2tune] <- MAP(fit)$parametersMAP
  CVfit[,i] <- pars_fit$def
  
  i = i+1
}

save(CVfit, file = "~/physics_guided_nn/data/Pmultisite_CVfit.Rdata")

gpp_train <- matrix(NA, nrow=nrow(allsites_train), ncol=length(unique(allsites_train$site)))
gpp_test <- matrix(NA, nrow=nrow(allsites_test), ncol=length(unique(allsites_train$site)))
et_train <- matrix(NA, nrow=nrow(allsites_train), ncol=length(unique(allsites_train$site)))
et_test <- matrix(NA, nrow=nrow(allsites_test), ncol=length(unique(allsites_train$site)))
sw_train <- matrix(NA, nrow=nrow(allsites_train), ncol=length(unique(allsites_train$site)))
sw_test <- matrix(NA, nrow=nrow(allsites_test), ncol=length(unique(allsites_train$site)))

load(file = "~/physics_guided_nn/data/Pmultisite_CVfit.Rdata")
i <- 1
for (s in unique(allsites_train$site)){
  
  load(file = paste0("~/physics_guided_nn/data/Pmultisite_fit_", s, ".Rdata"))
  
  gpp_train[,i] <- PRELES(PAR=allsites_train$PAR, TAir=allsites_train$Tair, VPD=allsites_train$VPD, Precip=allsites_train$Precip, CO2=allsites_train$CO2, fAPAR=allsites_train$fapar, p=CVfit[,i])$GPP
  gpp_test[,i] <- PRELES(PAR=allsites_test$PAR, TAir=allsites_test$Tair, VPD=allsites_test$VPD, Precip=allsites_test$Precip, CO2=allsites_test$CO2, fAPAR=allsites_test$fapar, p=CVfit[,i])$GPP
  
  et_train[,i] <- PRELES(PAR=allsites_train$PAR, TAir=allsites_train$Tair, VPD=allsites_train$VPD, Precip=allsites_train$Precip, CO2=allsites_train$CO2, fAPAR=allsites_train$fapar, p=CVfit[,i])$ET
  et_test[,i] <- PRELES(PAR=allsites_test$PAR, TAir=allsites_test$Tair, VPD=allsites_test$VPD, Precip=allsites_test$Precip, CO2=allsites_test$CO2, fAPAR=allsites_test$fapar, p=CVfit[,i])$ET
  
  sw_train[,i] <- PRELES(PAR=allsites_train$PAR, TAir=allsites_train$Tair, VPD=allsites_train$VPD, Precip=allsites_train$Precip, CO2=allsites_train$CO2, fAPAR=allsites_train$fapar, p=CVfit[,i])$SW
  sw_test[,i] <- PRELES(PAR=allsites_test$PAR, TAir=allsites_test$Tair, VPD=allsites_test$VPD, Precip=allsites_test$Precip, CO2=allsites_test$CO2, fAPAR=allsites_test$fapar, p=CVfit[,i])$SW
  
  i <- i+1
}

## Update data set with new calibrated Preles predictions ##

allsites_train$GPPp <- apply(gpp_train, 1, mean)
allsites_test$GPPp <- apply(gpp_test, 1, mean)
allsites_train$ETp <- apply(et_train, 1, mean)
allsites_test$ETp <- apply(et_test, 1, mean)
allsites_train$SWp <- apply(sw_train, 1, mean)
allsites_test$SWp <- apply(sw_test, 1, mean)

allsitesF <- rbind(allsites_train, allsites_test)

write.csv(allsitesF, file="~/physics_guided_nn/data/allsitesF.csv", row.names = FALSE)


pdf(file="~/physics_guided_nn/results/Pmultisitefit_BayesPriors.pdf", width=15, height=12)
par(mfrow=c(4, 4), mar=c(3,3,3,1))
for (i in 1:ncol(fit[[1]]$X)){ # loop over parameters fitted
  #fit1[[1]]$chain[1]
  # for DEzs:
  ests <- rbind(fit[[1]]$Z, fit[[2]]$Z, fit[[3]]$Z)
  plot(density(ests[,i], from=min(ests[,i]), to=max(ests[,i])), main=pars[pars2tune[i],1], las=1)
  abline(v=pars[pars2tune[i], 3:4], col="red")
}
dev.off()



## Generate files for prediction results ##

save(gpp_train, file = "~/physics_guided_nn/data/GPPp_multisite_train.Rdata")
save(gpp_test, file = "~/physics_guided_nn/data/GPPp_multisite_test.Rdata")


GPP_train <- apply(gpp_train, 1, mean)
GPP_test <- apply(gpp_test, 1, mean)
GPP_train_std <- apply(gpp_train, 1, sd)
GPP_test_std <- apply(gpp_test, 1, sd)

mae <- function(yhat, test=T){
  if (test){
    mae <- sum(abs(allsites_test$GPP - yhat))/length(yhat)
  }else{
    mae <- sum(abs(allsites_train$GPP - yhat))/length(yhat)
  }
  return(mae)
}
rmse <- function(yhat, test=T){
  if (test){
    rmse <- sqrt(sum((allsites_test$GPP - yhat)^2)/length(yhat))
  }else{
    rmse <- sqrt(sum((allsites_train$GPP - yhat)^2)/length(yhat))
  }
  return(rmse)
}

perfpormance_preles_full <- matrix(NA, nrow=length(unique(allsites_train$site)), ncol=4)
perfpormance_preles_full[,1] <- apply(gpp_train, 2, rmse, test=F)
perfpormance_preles_full[,2] <- apply(gpp_test, 2, rmse)
perfpormance_preles_full[,3] <- apply(gpp_train, 2, mae, test=F)
perfpormance_preles_full[,4] <- apply(gpp_test, 2, mae)

write.csv(perfpormance_preles_full, file="~/physics_guided_nn/results/performance_preles_multisite_full.csv")
write.csv(gpp_test, file="~/physics_guided_nn/results/preles_eval_preds_test_multisite_full.csv")
