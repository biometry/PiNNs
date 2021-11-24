#==================#
# Calibrate Preles #
#==================#
install.packages("lhs")

library(Rpreles)
library(lhs)
library(coda)
library(BayesianTools)

# Load default parameter values.
load("C:/Users/Niklas/Desktop/Uni/M.Sc. Environmental Science/Thesis/physics_guided_nn/data/parameterRanges.RData")
setwd("C:/Users/Niklas/Desktop/Uni/M.Sc. Environmental Science/Thesis/physics_guided_nn/data/")
parind <- c(5:10, 14:16) # Indexes for PRELES parameters

data <- read.csv("bily_kriz.csv")
eval <- data[startsWith(data$date, '2008'),]
train <- data[startsWith(data$date, "2001") | startsWith(data$date, "2002") | startsWith(data$date, "2004") | startsWith(data$date, "2006") | startsWith(data$date, "2007") ,]


#profound_out <- read.csv("~/Sc_Master/Masterthesis/Project/DomAdapt/data/profound/profound_out", sep=";")

#hyt = which(((profound_in$site == "hyytiala") & (profound_in$year %in% c(2001,2003,2004,2005,2006))))

#profound_in = profound_in[hyt,]
#profound_out = profound_out[hyt,]

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

# Check convergence:
tracePlot(chainDE, parametersOnly = TRUE, start = 1, whichParameters = 1:4)
tracePlot(chainDE, parametersOnly = TRUE, start = 1, whichParameters = 5:9)

marginalPlot(chainDE, scale = T, best = T, start = 5000)
correlationPlot(chainDE, parametersOnly = TRUE, start = 2000)

# save calibrated parameters
par$calib = par$def
par$calib[parind] = par.opt$parametersMAP
save(par, file = "bk_parameters.Rdata")

#==================#
# Check PERFORMANCE#
#==================#

library(Rpreles)


preds_train <- PRELES(TAir = train$Tair, PAR = train$PAR, VPD = train$VPD, Precip = train$Precip, fAPAR = train$fapar, CO2 = train$CO2,  p = par$calib[1:30])
preds_test <- PRELES(TAir = eval$Tair, PAR = eval$PAR, VPD = eval$VPD, Precip = eval$Precip, fAPAR = eval$fapar, CO2 = eval$CO2,  p = par$calib[1:30])
mse <- function(error)
{
  mean((error)^2)
}
mae <- function(e){mean(abs(e))}


plot(preds_test$GPP, type = "l")
points(eval$GPP)
mae(train$GPP-preds_train$GPP) # + (train$ET - preds_train$ET))
mae(eval$GPP-preds_test$GPP) #+ (eval$ET - preds_test$ET))

train['GPPp'] <- preds_train$GPP
train['ETp'] <- preds_train$ET
train['SWp'] <- preds_train$SW
write.csv(train, "train_bk.csv")

eval['GPPp'] <- preds_test$GPP
eval['ETp'] <- preds_test$ET
eval['SWp'] <- preds_test$SW
write.csv(eval, "test_bk.csv")




load("C:/Users/Niklas/Desktop/Uni/M.Sc. Environmental Science/Thesis/physics_guided_nn/data/hyytiala_parameters.RData")
test_hyt <- read.csv('test_hyt.csv')
train_hyt <- read.csv('train_hyt.csv')

ts <- PRELES(TAir = train$Tair, PAR = train$PAR, VPD = train$VPD, Precip = train$Precip, fAPAR = train$fapar, CO2 = traint$CO2,  p = par$calib[1:30])
tes <- PRELES(TAir = test_hyt$Tair, PAR = test$PAR, VPD = testt$VPD, Precip = testt$Precip, fAPAR = tes$fapar, CO2 = test$CO2,  p = par$calib[1:30])
train_hyt['ETp'] <- ts$ET
train_hyt['SWp'] <- ts$SW
write.csv(train_hyt, 'train_hyt.csv')

test_hyt['ETp'] <- tes$ET
test_hyt['SWp'] <- tes$SW
write.csv(test_hyt, 'test_hyt.csv')

round(test_hyt$GPPp, 3) == round(tes$GPP, 3)



