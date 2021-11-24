# Cross Validation PRELES
install.packages("lhs")

library(Rpreles)
library(lhs)
library(coda)
library(BayesianTools)

# Load default parameter values.
setwd("C:/Users/Niklas/Desktop/Uni/M.Sc. Environmental Science/Thesis/physics_guided_nn/data/")
data <- read.csv("hyytiala.csv")
eval <- data[startsWith(data$date, '2008'),]
train <- data[!startsWith(data$date, "2008") & !startsWith(data$date, "2007"),]
rmse <- function(error){sqrt(mean((error)^2))}
mae <- function(e){mean(abs(e))}

### START CV
years <- unique(format(as.Date(train$date), format="%Y"))
nfolds <- length(years)
#preds_train <- matrix(ncol=nfolds, nrow=length(eval$date))
preds_test <- matrix(ncol=nfolds, nrow=length(eval$date))
performance <- matrix(nrow = nfolds, ncol = 4)

for (fold in (1:nfolds)){
  print(fold)
  load("C:/Users/Niklas/Desktop/Uni/M.Sc. Environmental Science/Thesis/physics_guided_nn/data/parameterRanges.RData")
  parind <- c(5:10, 14:16) # Indexes for PRELES parameters
  train = train[!startsWith(train$date, years[fold]),]
  
  #1- Likelihood function
  likelihood <- function(pValues){
    p <- par$def
    p[parind] <- pValues # new parameter values
    predicted<- PRELES(DOY=train$DOY,PAR=train$PAR,TAir=train$Tair,VPD=train$VPD,Precip=train$Precip,
                     CO2=train$CO2,fAPAR=train$fapar,p=p[1:30])
    diff_GPP <- predicted$GPP-train$GPP
    #diff_ET <- predicted$ET-train$ET
    # mäkälä
    llvalues <- sum(dnorm(predicted$GPP, mean = train$GPP, sd = p[31], log=T))
    ###   llvalues <- sum(dnorm(diff_GPP, sd = p[31], log=T))
    #llvalues <- sum(dexp(abs(diff_GPP),rate = 1/(p[31]+p[32]*predicted$GPP),log=T))
    return(llvalues)
  }

  #2- Prior
  prior <- createUniformPrior(lower = par$min[parind], upper = par$max[parind])

  #=Bayesian set up=#

  BSpreles <- createBayesianSetup(likelihood, prior, best = par$def[parind], names = par$name[parind], parallel = F)

  bssetup <- checkBayesianSetup(BSpreles)

  #=Run the MCMC with three chains=#

  settings <- data.frame(iterations = 1e5, optimize=F, nrChains = 3)
  chainDE <- runMCMC(BSpreles, sampler="DEzs", settings = settings)
  par.opt<-MAP(chainDE) #gets the optimized maximum value for the parameters




  # save calibrated parameters
  par$calib = par$def
  par$calib[parind] = par.opt$parametersMAP
  save(par, file = sprintf("parameters_hyt_fold_%s.Rdata", fold))



  ptr <- PRELES(TAir = train$Tair, PAR = train$PAR, VPD = train$VPD, Precip = train$Precip, fAPAR = train$fapar, CO2 = train$CO2,  p = par$calib[1:30], returncols = c("GPP"))
  pte <- PRELES(TAir = eval$Tair, PAR = eval$PAR, VPD = eval$VPD, Precip = eval$Precip, fAPAR = eval$fapar, CO2 = eval$CO2,  p = par$calib[1:30], returncols = c("GPP"))
  
  performance[fold,1] <- rmse(train$GPP-ptr$GPP) # + (train$ET - preds_train$ET))
  performance[fold,2] <- rmse(eval$GPP-pte$GPP) #+ (eval$ET - preds_test$ET))
  performance[fold,3] <- mae(train$GPP-ptr$GPP) # + (train$ET - preds_train$ET))
  performance[fold,4] <- mae(eval$GPP-pte$GPP) #+ (eval$ET - preds_test$ET))
  
  #preds_train[,fold] <- ptr$GPP
  preds_test[,fold] <- pte$GPP
  data <- read.csv("hyytiala.csv")
  train <- data[!startsWith(data$date, "2008") & !startsWith(data$date, "2007"),]

}

perf_df <- as.data.frame(performance)
colnames(perf_df) <- c("train_RMSE", "test_RMSE", "train_MAE", "test_MAE")
write.csv(perf_df, "performancePRELES.csv")

write.csv(preds_train, "p_train_hyt.csv")
write.csv(preds_test, "p_test_hyt.csv")


