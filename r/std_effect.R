library(Rpreles)
library(BayesianTools)

setwd("~/PycharmProjects/PiNNs")

source("r/helpers.R")

load("data/parameterRanges.rdata") # parameter defaults/ranges
# par # note that "-999" is supposed to indiate NA!
pararms <- par # unfortunate naming "par" replaced by "pararms"
rm(par)
pararms[pararms=="-999"] <- NA
pararms # note that some parameters are set without uncertainty (e.g. soildepth)

# jetzt neu: S[max]
#pars[pars$name=="S[max]", 4] <- 45 # was 30
#  S[max]: tick
pararms[pararms$name=="nu", 4] <- 10 # was 5


# select the parameters to be calibrated:
pars2tune <- c(5:11, 14:18, 31) # note that we omit 32, as it refers to ET
thispar <- pararms$def
names(thispar) <- pararms$name

hyytiala <- read.csv("data/hyytiala.csv")
hyytiala$date <- as.Date(hyytiala$date)
hyytiala$year <- format(hyytiala$date, format="%Y")

hyytiala_train <- hyytiala[!(hyytiala$year %in% c("2008", "2007", "2005", "2004")), ]
hyytiala_test <- hyytiala[hyytiala$year == "2008", ]

results <- matrix(nrow = 15, ncol = 3)
results[,1] <- seq(from =1, to = 5, length.out =15)

for (i in c(1:15)){
  
  ell <- function(pars, data=hyytiala_train){
    # pars is a vector the same length as pars2tune
    thispar[pars2tune] <- pars
    # likelihood function, first shot: normal density
    with(data, sum(dnorm(hyytiala_train$GPP, mean=PRELES(PAR=hyytiala_train$PAR, 
                                               TAir=hyytiala_train$Tair, 
                                               VPD=hyytiala_train$VPD, 
                                               Precip=hyytiala_train$Precip, 
                                               CO2=hyytiala_train$CO2, 
                                               fAPAR=hyytiala_train$fapar , 
                                               p=thispar)$GPP, sd=results[i,1], log=T)))
  }
  priors <- createUniformPrior(lower=pararms$min[pars2tune], upper=pararms$max[pars2tune], best=pararms$def[pars2tune])
  setup <- createBayesianSetup(likelihood=ell, prior=priors, parallel=T)
  settings <- list(iterations=50000, adapt=T, nrChains=3, parallel=T) # runs 3 chains in parallel for each chain ...
  # run:
  fit <- runMCMC(bayesianSetup = setup, settings = settings, sampler = "DREAMzs")
  
  pars_fit <- pararms
  pars_fit$def[pars2tune] <- MAP(fit)$parametersMAP
  
  
  gpp_preds <- PRELES(PAR=hyytiala_test$PAR, TAir=hyytiala_test$Tair, VPD=hyytiala_test$VPD, Precip=hyytiala_test$Precip, CO2=hyytiala_test$CO2, fAPAR=hyytiala_test$fapar, p=pars_fit$def)$GPP
  
  rmse <- function(x, x_hat){
    return(sqrt(mean((x-x_hat)**2)))
  }
  mae <- function(x, x_hat){
    return(mean(abs(x-x_hat)))
  }
  
  results[i,2] <- rmse(gpp_preds, hyytiala_test$GPP)
  results[i,3] <- mae(gpp_preds, hyytiala_test$GPP)

}

plot( hyytiala_test$GPP, type='l')
lines(gpp_preds, col="red")

summary(lm(results[,3] ~ results[,1]))
summary(lm(results[,2] ~ results[,1]))
plot(lm(results[,2] ~ results[,1]))

# Fit the linear models
lm1 <- lm(results[,2] ~ results[,1])
lm2 <- lm(results[,3] ~ results[,1])

# Extract the first coefficients
coef1 <- coef(lm1)[2]
coef2 <- coef(lm2)[2]

par(mfrow=c(1, 2))
plot(round(results[,1], 2), results[,2], type="l", ylab = "RMSE", xlab = "Standard dev. in BC")
#abline(lm1, col="red")
plot(round(results[,1], 2), results[,3], type="l", ylab = "MAE", xlab = "Standard dev. in BC")
#abline(lm2, col="red")

# ======================= #
# Redo PRELES fit with sd #
# ======================= #


ell <- function(pars, data=hyytiala_train){
  # pars is a vector the same length as pars2tune
  thispar[pars2tune] <- pars
  # likelihood function, first shot: normal density
  with(data, sum(dnorm(hyytiala_train$GPP, mean=PRELES(PAR=hyytiala_train$PAR, 
                                                       TAir=hyytiala_train$Tair, 
                                                       VPD=hyytiala_train$VPD, 
                                                       Precip=hyytiala_train$Precip, 
                                                       CO2=hyytiala_train$CO2, 
                                                       fAPAR=hyytiala_train$fapar , 
                                                       p=thispar)$GPP, sd=1, log=T)))
}

ell_sd_fit <- function(pars, data=hyytiala_train){
  # pars is a vector the same length as pars2tune
  thispar[pars2tune] <- pars
  # likelihood function, first shot: normal density
  with(data, sum(dnorm(hyytiala_train$GPP, mean=PRELES(PAR=hyytiala_train$PAR, 
                                                       TAir=hyytiala_train$Tair, 
                                                       VPD=hyytiala_train$VPD, 
                                                       Precip=hyytiala_train$Precip, 
                                                       CO2=hyytiala_train$CO2, 
                                                       fAPAR=hyytiala_train$fapar , 
                                                       p=thispar)$GPP, sd=pars[length(pars)], log=T)))
}
priors <- createUniformPrior(lower=pararms$min[pars2tune], upper=pararms$max[pars2tune], best=pararms$def[pars2tune])
setup <- createBayesianSetup(likelihood=ell, prior=priors, parallel=T)
settings <- list(iterations=50000, adapt=T, nrChains=3, parallel=T) # runs 3 chains in parallel for each chain ...
# run:
fit <- runMCMC(bayesianSetup = setup, settings = settings, sampler = "DREAMzs")

pars_fit <- pararms
pars_fit$def[pars2tune] <- MAP(fit)$parametersMAP


gpp_preds <- PRELES(PAR=hyytiala_test$PAR, TAir=hyytiala_test$Tair, VPD=hyytiala_test$VPD, Precip=hyytiala_test$Precip, CO2=hyytiala_test$CO2, fAPAR=hyytiala_test$fapar, p=pars_fit$def)$GPP


pdf(file=paste0("/Users/mw1205/Library/Mobile Documents/com~apple~CloudDocs/Projects/physics_guided_nn_meta/manuscript/revisions/PRELES_no_sd_fit.pdf"), width=15, height=12)
par(mfrow=c(4, 4), mar=c(3,3,3,1))
for (i in 1:ncol(fit[[1]]$Z)){ # loop over parameters fitted
  plot(density(fit[[1]]$Z[,i]), from=min(fit[[1]]$Z[,i]), to=max(fit[[1]]$Z[,i]), main=pararms[pars2tune[i],1], las=1)
  abline(v=pararms[pars2tune[i], 3:4], col="red")
  text(x=min(fit[[1]]$Z[,i]), y=max(density(fit[[1]]$Z[,i])$y), labels=round(MAP(fit)$parametersMAP[i],2), pos=4, cex=1.2) # Add number in upper left corner
}
dev.off()
