#==========================#
# VIA PRELES / conditional #
#==========================#

library(Rpreles)

predict <- function(x_test, data_use){
  
  CVfit = read.csv(paste0("~/Projects/physics_guided_nn/data/Psinglesite_CVfit_", data_use, ".csv"))[,2:6]
  
  preds = matrix(nrow = nrow(x_test), ncol=ncol(CVfit))
  
  for (i in 1:ncol(CVfit)){
    preds[,i] = PRELES(PAR=x_test$PAR, TAir=x_test$Tair, VPD=x_test$VPD, Precip=x_test$Precip, CO2=x_test$CO2, fAPAR=x_test$fapar, p=CVfit[,i])$GPP
  }
  
  preds = apply(preds, 1, mean)
  print(preds)
  
  return(preds)
}


via <- function(data_use, gridsize = 200){

  hyytiala <- read.csv("~/Projects/physics_guided_nn/data/hyytiala.csv")
  hyytiala$date <- as.Date(hyytiala$date)
  hyytiala$year <- format(hyytiala$date, format="%Y")
  
  hyytiala_train <- hyytiala[!(hyytiala$year %in% c("2008", "2007", "2005", "2004")), ]
  hyytiala_test <- hyytiala[hyytiala$year == "2008", ]
  hyytiala_test <- hyytiala_test[2:nrow(hyytiala_test),] # remove first day of year (Niklas, why?)
  
  variables = c('PAR', 'Tair', 'VPD', 'Precip', 'fapar')
  
  thresholds = data.frame('PAR' = c(0, 200), 'Tair'= c(-20, 40), 'VPD' = c(0, 60), 'Precip'= c(0, 100), 'fapar'= c(0, 1))

  for (v in variables){
    
    var_range = seq(thresholds[v][1,], thresholds[v][2,], length.out=gridsize)
    day_names = c("mar", "jun", "sep", "dec")
    x_test = hyytiala_test
    
    mar = with(hyytiala_test, hyytiala_test[(date >= "2008-03-13" & date <= "2008-03-27"),])[,c(3:11)]
    #mar = data.frame(as.list(colMeans(mar)))[rep(1, times=gridsize),]
    #mar = data.frame(mar)
    jun = with(hyytiala_test, hyytiala_test[(date >= "2008-06-14" & date <= "2008-06-28"),])[,c(3:11)]
    #jun = data.frame(as.list(colMeans(jun)))[rep(1, times=gridsize),]
    sep = with(hyytiala_test, hyytiala_test[(date >= "2008-09-13" & date <= "2008-09-27"),])[,c(3:11)]
    #sep = data.frame(as.list(colMeans(sep)))[rep(1, times=gridsize),]
    dec = with(hyytiala_test, hyytiala_test[(date >= "2008-12-14" & date <= "2008-12-28"),])[,c(3:11)]
    #dec = data.frame(as.list(colMeans(dec)))[rep(1, times=gridsize),]

    days = list(mar, jun, sep, dec)
    
    n = 1
    for (day in days){
      print(nrow(day))
      output = matrix(nrow = length(var_range), ncol = 15)
      for (j in 1:length(var_range)){
        day[v] = var_range[j]
        preds = predict(day, data_use)
        print(j)
        output[j,] = preds
      }
      write.csv(output, paste0("~/Projects/physics_guided_nn/results/preles_", data_use, "_", v, "_via_cond_", day_names[n], ".csv"))
      n = n+1
    }
    #output = cbind(var_range, output)

  }
  return(output)
}


out_full = via("full", gridsize = 200)
#out_sparse = via("sparse")
