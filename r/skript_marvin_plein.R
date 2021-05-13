
## Load packages:
library(neuralnet)
library(Rpreles)
library(BayesianTools)
library(sensitivity)
library(coda)
library(hydroGOF)


## Clear workspace and set seed for reproducibility:
rm(list = ls())
set.seed(501)

## Set workspace and load data:
setwd("C:/Users/Marvin/Desktop/Modelling Environmental Systems/PReLES")
load("C:/Users/Marvin/Desktop/Modelling Environmental Systems/PReLES/EddyCovarianceDataBorealSites.rdata")
load("C:/Users/Marvin/Desktop/Modelling Environmental Systems/PReLES/parameterRanges.rdata")


##
##
## Sensitivity analysis
##
##

## Local:
defaults = c(413, 0.45, 0.118, 3, 0.7457, 10.93, -3.063,
             17.72, -0.1027, 0.03673, 0.7779, 0.5, -0.364, 0.2715,
             0.8351, 0.07348, 0.9996, 0.4428, 1.2, 0.33, 4.970496,
             0, 0, 160, 0, 0, 0, -999, -999, -999)

parms <- data.frame(best = defaults)


parms$best[parms$best == -999] <- NA
parms$lower <- parms$best-abs(0.1*parms$best)
parms$upper <- parms$best+abs(0.1*parms$best)


sensitivityTarget <- function(parm){
  par_temp <- parms
  results <- numeric(3)
  names(results) <- c("best", "lower", "upper")
  for( i in 1:3){
    par_temp$best[parm] <- par_temp[parm,i]
    predicted <- PRELES(PAR=s1$PAR, TAir=s1$TAir, VPD=s1$VPD, 
                        Precip=s1$Precip, CO2=s1$CO2, fAPAR=s1$fAPAR, p = par_temp$best)$GPP
    results[i] <- mean(predicted)
  }
  return(results)
}




sens_data <- data.frame(best = numeric(nrow(parms)),
                        lower = numeric(nrow(parms)),
                        upper = numeric(nrow(parms)))

# sensitivityTarget()
for(j in 1:nrow(parms)){
  sens_data[j,] <- sensitivityTarget(j)
}

rownames(sens_data) <- rownames(parms)

## Calculate percentual change:
sens_data$percentual_lower <- round(sens_data$lower/sens_data$best*100-100, 2)
sens_data$percentual_upper <- round(sens_data$upper/sens_data$best*100-100, 2)

sens_data <- cbind(rownames(sens_data),parms, sens_data)

## Create plots for the GPP-Parameters for which there are minima and maxima.

## Try with all parameters that have min and max values or resulted in change above:
sel_par <- c(5:11, 14, 31, 32)


par(mfrow = c(3,4))
for(i in sel_par){
  par_temp <- parms
  seq_temp <- seq(from = parms$min[i], parms$max[i], len = 20)
  results <- numeric(length(seq_temp))
  for(j in 1:length(seq_temp)){
    par_temp$def[i] <- seq_temp[j]
    predicted <- PRELES(PAR=s1$PAR, TAir=s1$TAir, VPD=s1$VPD, 
                        Precip=s1$Precip, CO2=s1$CO2, fAPAR=s1$fAPAR, p = par_temp$def)$GPP
    results[j] <- mean(predicted)
  }
  plot(seq_temp, results, xlab = par_temp$name[i], ylab = "mean GPP", type = "o", ylim = c(0,5))
  abline(v = par$def[i], col = "red")
}



 ## Delete parameters 31 and 32 as they have no effect:

sel_par <- c(5:11, 14)

par(mfrow = c(2,4))
for(i in sel_par){
  par_temp <- par
  seq_temp <- seq(from = par$min[i], par$max[i], len = 20)
  results <- numeric(length(seq_temp))
  for(j in 1:length(seq_temp)){
    par_temp$def[i] <- seq_temp[j]
    predicted <- PRELES(PAR=s1$PAR, TAir=s1$TAir, VPD=s1$VPD, 
                        Precip=s1$Precip, CO2=s1$CO2, fAPAR=s1$fAPAR, p = par_temp$def)$GPP
    results[j] <- mean(predicted)
  }
  plot(seq_temp, results, xlab = par_temp$name[i], ylab = "mean GPP", type = "o", ylim = c(0,5))
  abline(v = par$def[i], col = "red")
}



par_sel <- c(5:11, 14:18, 31:32)
sensitivityTarget2 <- function(mat){
  result <- numeric(nrow(mat))
  for(i in 1:nrow(mat)){
    par_temp <- par2$def
    par_temp[par_sel] <- as.vector(mat[i,])
    predicted <- PRELES(PAR=s1$PAR, TAir=s1$TAir, VPD=s1$VPD, 
                        Precip=s1$Precip, CO2=s1$CO2, fAPAR=s1$fAPAR, p = par_temp)$GPP
    result[i] <- mean(predicted)
  }
  return(result)
}


## Global analysis using morris screening:
par2 <- par

par2[which(par2 == -999, arr.ind = T)] <- NA

par(mfrow=c(1,1))
morrisOut <- morris(model=sensitivityTarget2, factors=par2$name[par_sel],
                    r=500, design= list(type="oat",levels=10, grid.jump=2),
                    binf=par2$min[par_sel],
                    bsup=par2$max[par_sel], scale=TRUE)

## Plotting morris:
oldpar <- par()
par(xpd=T, las=1)
plot(morrisOut)
par(oldpar)


##
##
## Calibration:
##
##


parSel <- c(5:11, 14:18, 31,32) 

par2$min[31:32] <- -5
par2$max[31:32] <- 5

#optim function for morris and MCMC
opt_fun <- 
  function(par,sum =TRUE, distr = "dnorm"){
    parms <- par2$def
    parms[parSel] <- par
    predicted <- PRELES(PAR=s1$PAR, TAir=s1$TAir, VPD=s1$VPD,Precip=s1$Precip, CO2=s1$CO2, 
                        fAPAR=s1$fAPAR, p = parms)
    if(distr == "dnorm"){
      llValues <- dnorm(s1$GPPobs,mean = predicted$GPP, sd=exp(parms[31] + parms[32]*predicted$GPP),log=T)}
    else{diff_GPP <- predicted$GPP-s1$GPPobs
    llValues <- dexp(abs(diff_GPP),rate = 1/(parms[31]+parms[32]*predicted$GPP),log=T)}
    if(sum == FALSE)
      return(llValues)
    else
      return(sum(llValues))
  }


prior <- createUniformPrior(lower = par2$min[parSel] , upper =  par2$max[parSel], best = par2$def[parSel])
bayesianSetup_multi <- createBayesianSetup(likelihood = opt_fun, prior = prior, names = par2$name[parSel])


## run MCMC:
settings <- list(iterations=100000, nrChains=1)
out.mcmc <- runMCMC(bayesianSetup=bayesianSetup_multi,sampler=  "DEzs",settings=settings)
summary(out.mcmc)


par(mfrow = c(5,5))

plot(out.mcmc)

par(oldpar)
Sample <- getSample(out.mcmc,start=5000, end = 0.33*settings$iterations, coda=T, thin = 10)
p <- par2$def
nSample <- nrow(Sample[[1]])
GPPerror <- GPPsim <- matrix(0, nrow = nSample, ncol = length(s1$GPPobs))
for (i in 1:nrow(Sample[[1]])){
  p[parSel] <- Sample[[1]][i, ]
  preles_s1 <- PRELES(DOY=s1$DOY,PAR=s1$PAR, TAir=s1$TAir, VPD=s1$VPD,
                      Precip=s1$Precip, CO2=s1$CO2, fAPAR=s1$fAPAR, p=p[1:30])
  GPPsim[i, ] <- preles_s1$GPP
  GPPerror[i, ] <- rnorm(length(preles_s1$GPP), mean = 0, sd = exp(p[31] + p[32]*preles_s1$GPP))
}


## Plot time series:
plotTimeSeries(s1$GPPobs, colMeans(GPPsim),
               confidenceBand = apply(GPPsim, 2, quantile, probs = c(0.025, 0.975)),
               predictionBand = apply(GPPsim + GPPerror, 2, quantile, probs = c(0.025, 0.975)))



# Create additional plots for diagnostics:
correlationPlot(out.mcmc)
gelmanDiagnostics(out.mcmc, plot = T)

par(mfrow = c(3,4))
cumuplot(out.mcmc$chain)


#########################################
##Validation on site not used for fitting
#########################################
# Idea: Use the opt_function to get the likelihood 
## Predict with the optimised parameter values and the default parameter values for other sites and compare the loglikelihood.
#optim function for morris and MCMC
opt_fun_2 <- 
  function(par,sum =TRUE, distr = "dnorm", site, optimised_parms = F){
    parms <- par2$def
    if(optimised_parms == T){
      parms[parSel] <- par}
    predicted <- PRELES(PAR=site$PAR, TAir=site$TAir, VPD=site$VPD,Precip=site$Precip, CO2=site$CO2, 
                        fAPAR=site$fAPAR, p = parms)
    if(distr == "dnorm"){
      llValues <- dnorm(site$GPPobs,mean = predicted$GPP, sd=exp(parms[31] + parms[32]*predicted$GPP),log=T)}
    else{diff_GPP <- predicted$GPP-s1$GPPobs
    llValues <- dexp(abs(diff_GPP),rate = 1/(parms[31]+parms[32]*predicted$GPP),log=T)}
    if(sum == FALSE)
      return(llValues)
    else
      return(sum(llValues, na.rm = T))
  }


optim_par <- MAP(out.mcmc)$parametersMAP
# test for site 1:
opt_fun_2(par = optim_par, sum = T, site = s1, optimised_parms = T)
opt_fun_2(par = optim_par, sum = T, site = s1, optimised_parms = F)

# Now for the other sites:
opt_fun_2(par = optim_par, sum = T, site = s2, optimised_parms = T)
opt_fun_2(par = optim_par, sum = T, site = s2, optimised_parms = F)

opt_fun_2(par = optim_par, sum = T, site = s3, optimised_parms = T)
opt_fun_2(par = optim_par, sum = T, site = s3, optimised_parms = F)

opt_fun_2(par = optim_par, sum = T, site = s4, optimised_parms = T)
opt_fun_2(par = optim_par, sum = T, site = s4, optimised_parms = F)


##
##
## Prediction uncertainty - Predict to other site (s2), calculate prediction uncertainty and compare to neural network:
##
##

Sample2 <- getSample(out.mcmc,start=5000, end = 0.33*settings$iterations, coda=T, thin = 10)
p <- par2$def
nSample <- nrow(Sample2[[1]])
GPPerror2 <- GPPsim2 <- matrix(0, nrow = nSample, ncol = length(s1$GPPobs))
for (i in 1:nrow(Sample2[[1]])){
  p[parSel] <- Sample2[[1]][i, ]
  preles_s2 <- PRELES(DOY=s2$DOY,PAR=s2$PAR, TAir=s2$TAir, VPD=s2$VPD,
                      Precip=s2$Precip, CO2=s2$CO2, fAPAR=s2$fAPAR, p=p[1:30])
  GPPsim2[i, ] <- preles_s2$GPP
  GPPerror2[i, ] <- rnorm(length(preles_s2$GPP), mean = 0, sd = exp(p[31] + p[32]*preles_s2$GPP))
}




### Run neural networks:
data <- rbind(s1,s2)
data <- data[,c("PAR","TAir","VPD","Precip", "DOY", "GPPobs")]
apply(data,2,function(x) sum(is.na(x)))
index <- 1:(nrow(data)/2)
train <- data[index,]
test <- data[-index,]


maxs <- apply(data, 2, max) 
mins <- apply(data, 2, min)
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))
train_ <- scaled[index,]
test_ <- scaled[-index,]


n <- names(train_)
f <- as.formula(paste("GPPobs ~", paste(n[!n %in% c("GPPobs", "ETobs, CO2", "ETobs")], collapse = " + ")))
nn <- neuralnet(f,data=train_,hidden=c(7,5),linear.output=F)




x <- train_[,-6]
y <- train_[,6]
newData <- train_[,-6]
## Get prediction intervals:
yPredInt_train <- nnetPredInt(nn, x, y, newData)
## Rescale predictions:
yPredInt_rescaled_train <- apply(yPredInt_train, 2, function(x) x*(max(data$GPPobs)-min(data$GPPobs))+min(data$GPPobs))



## Get predictions and intervals for site 1 training data:
x <- train_[,-6]
y <- train_[,6]
newData <- test_[,-6]
yPredInt <- nnetPredInt(nn, x, y, newData)


## Rescale:
yPredInt_rescaled <- apply(yPredInt, 2, function(x) x*(max(data$GPPobs)-min(data$GPPobs))+min(data$GPPobs))





## Plots:

# pdf("net.pdf", width = 6.95, height = 6.61)
par(mfrow = c(2,2), mar = c(5.1, 5, 4.1, 2.1))

## For PRELES and site 1:
predictionBand <- apply(GPPsim + GPPerror, 2, quantile, probs = c(0.025, 0.975))
plot(1:730, colMeans(GPPsim), type = "l", col = "red", ylim = c(0,10), xlab = "Day of the year", ylab = expression('GPP [g C m'^-2*'day'^-1*']'), main = "PRELES at site 1")

points(s1$GPPobs, pch = 3, cex = 0.6, col = "black")
polygon(c(1:730, rev(1:730)), c(predictionBand[2,], rev(predictionBand[1,])),
        col=rgb(1, 0, 0,0.5), border = NA)

## For neural networks and site 1:
plot(1:730, yPredInt_rescaled_train[,1], type = "l", col = "red", ylim = c(0,10), main = "Neural network at site 1", ylab = "", xlab = "Day of the year")

points(train$GPPobs, pch = 3, cex = 0.6, col = "black")
polygon(c(1:730, rev(1:730)), c(yPredInt_rescaled_train[,3], rev(yPredInt_rescaled_train[,2])),
        col=rgb(1, 0, 0,0.5), border = NA)



## For PRELES and site 2:
predictionBand <- apply(GPPsim2 + GPPerror2, 2, quantile, probs = c(0.025, 0.975))
plot(1:730, colMeans(GPPsim2), type = "l", col = "red", ylim = c(0,10), xlab = "Day of the year", ylab = expression('GPP [g C m'^-2*'day'^-1*']'), main = "PRELES at site 2")

points(s2$GPPobs, pch = 3, cex = 0.6, col = "black")
polygon(c(1:730, rev(1:730)), c(predictionBand[2,], rev(predictionBand[1,])),
        col=rgb(1, 0, 0,0.5), border = NA)

## For neural networks and site 2:
plot(1:730, yPredInt_rescaled[,1], type = "l", col = "red", ylim = c(0,10), main = "Neural network at site 2", ylab = "", xlab = "Day of the year")

points(test$GPPobs, pch = 3, cex = 0.6, col = "black")
polygon(c(1:730, rev(1:730)), c(yPredInt_rescaled[,3], rev(yPredInt_rescaled[,2])),
        col=rgb(1, 0, 0,0.5), border = NA)
# dev.off()



## Calculate rmses:

rmse(colMeans(GPPsim), s1$GPPobs)
rmse(yPredInt_rescaled_train[,1], s1$GPPobs)

rmse(colMeans(GPPsim2), s2$GPPobs)
rmse(yPredInt_rescaled[,1], s2$GPPobs)

