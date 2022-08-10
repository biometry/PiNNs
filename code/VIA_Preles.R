#============#
# VIA PRELES #
#============#

library(Rpreles)

predict <- function(x_test, data_use){
  
  CVfit = read.csv(paste0("~/Projects/physics_guided_nn/data/Psinglesite_CVfit_", data_use, ".csv"))[,2:6]
  
  preds = matrix(nrow = nrow(x_test), ncol=ncol(CVfit))
  
  for (i in 1:ncol(CVfit)){
    preds[,i] = PRELES(PAR=x_test$PAR, TAir=x_test$Tair, VPD=x_test$VPD, Precip=x_test$Precip, CO2=x_test$CO2, fAPAR=x_test$fapar, p=CVfit[,i])$GPP
  }
  
  return(preds)
}


via <- function(data_use, gridsize = 200){

  hyytiala <- read.csv("~/Projects/physics_guided_nn/data/hyytiala.csv")
  hyytiala$date <- as.Date(hyytiala$date)
  hyytiala$year <- format(hyytiala$date, format="%Y")
  
  hyytiala_train <- hyytiala[!(hyytiala$year %in% c("2008", "2007", "2005", "2004")), ]
  hyytiala_test <- hyytiala[hyytiala$year == "2008", ]
  
  variables = c('PAR', 'Tair', 'VPD', 'Precip', 'fapar')
  
  thresholds = data.frame('PAR' = c(0, 200), 'Tair'= c(-20, 40), 'VPD' = c(0, 60), 'Precip'= c(0, 100), 'fapar'= c(0, 1))

  for (v in variables){
    
    var_range = seq(thresholds[v][1,], thresholds[v][2,], length.out=gridsize)
    output = matrix(nrow = gridsize, ncol = length(hyytiala_test$date))
    
    x_test = hyytiala_test
    
    for (i in 1:gridsize){
      x_test[v] = var_range[i]
      preds = apply(predict(x_test, data_use), 1, mean)
      output[i,] = preds
    }
    write.csv(output, paste0("~/Projects/physics_guided_nn/results/preles_", data_use, "_", v, "_via.csv"))
  }
}

via("full")
via("sparse")
