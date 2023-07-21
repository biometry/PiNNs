#==========================#
# VIA PRELES / conditional #
#==========================#

library(Rpreles)
data_use = 'full'
prediction_scenario = 'spatial'
gridsize=200

predict_via <- function(x_test, data_use, prediction_scenario){
  
  if (prediction_scenario =='temporal'){
    CVfit = read.csv(paste0("~/PycharmProjects/physics_guided_nn/data/Psinglesite_CVfit_", data_use, ".csv"))#[,2:6]
  }else{
    CVfit = read.csv(paste0("~/PycharmProjects/physics_guided_nn/data/Pmultisite_CVfit_", data_use, ".csv"))[,2:6]
  }
  
  if (prediction_scenario == 'temporal'){
    preds = matrix(nrow = nrow(x_test), ncol=ncol(CVfit))
    for (i in 1:ncol(CVfit)){
      preds[,i] = PRELES(PAR=x_test$PAR, TAir=x_test$Tair, VPD=x_test$VPD, Precip=x_test$Precip, CO2=x_test$CO2, fAPAR=x_test$fapar, p=CVfit[,i])$GPP
    }
    preds = apply(preds, 1, mean)
  }else{
    sites <- unique(x_test$X)
    preds_sitewise <- list()
    for (site in sites){
      df <- x_test[x_test$X==site,]
      preds <- matrix(nrow = nrow(df), ncol=ncol(CVfit))
      for (i in 1:ncol(CVfit)){
        preds[,i] = PRELES(PAR=df$PAR, TAir=df$Tair, VPD=df$VPD, Precip=df$Precip, CO2=df$CO2, fAPAR=df$fapar, p=CVfit[,i])$GPP
      }
      preds <- apply(preds, 1, mean)
      preds_sitewise[[site]] <- preds
    }
  }
  
  return(preds)
}


via <- function(data_use, prediction_scenario, gridsize = 200){
  
  if (prediction_scenario == 'temporal'){
    if (data_use == "sparse"){
      hyytiala <- read.csv("~/PycharmProjects/physics_guided_nn/data/hyytialaF_sparse.csv")
    }else{
      hyytiala <- read.csv("~/PycharmProjects/physics_guided_nn/data/hyytialaF_full.csv")
    }
  }else{
    if (data_use == "sparse"){
      hyytiala <- read.csv("~/PycharmProjects/physics_guided_nn/data/allsitesF_sparse.csv")
    }else{
      hyytiala <- read.csv("~/PycharmProjects/physics_guided_nn/data/allsitesF_full.csv")
    }
  }
  
  
  hyytiala$date <- as.Date(hyytiala$date)
  hyytiala$year <- format(hyytiala$date, format="%Y")
  
  hyytiala_train <- hyytiala[!(hyytiala$year %in% c("2008", "2007", "2005", "2004")), ]
  hyytiala_test <- hyytiala[hyytiala$year == "2008", ]
  hyytiala_test <- hyytiala_test[2:nrow(hyytiala_test),] # remove first day of year (Niklas, why?)
  
  variables = c('PAR', 'Tair', 'VPD', 'Precip', 'fapar')
  
  thresholds = data.frame('PAR' = c(0, 200), 'Tair'= c(-20, 40), 'VPD' = c(0, 60), 'Precip'= c(0, 100), 'fapar'= c(0, 1))
  v = 'PAR'
  
  for (v in variables){
    
    var_range = seq(thresholds[v][1,], thresholds[v][2,], length.out=gridsize)
    day_names = c("mar", "jun", "sep", "dec")
    x_test = hyytiala_test
    x_test$X = gsub("[^a-zA-Z]", "", x_test$X)
    
    mar = with(x_test, x_test[(date >= "2008-03-13" & date <= "2008-03-27"),])[,c(3:13)]
    jun = with(x_test, x_test[(date >= "2008-06-14" & date <= "2008-06-28"),])[,c(3:13)]
    sep = with(x_test, x_test[(date >= "2008-09-13" & date <= "2008-09-27"),])[,c(3:13)]
    dec = with(x_test, x_test[(date >= "2008-12-14" & date <= "2008-12-28"),])[,c(3:13)]
  
    days = list(mar, jun, sep, dec)
    
    n = 1
    for (day in days){
      day = day
      output = matrix(nrow = length(var_range), ncol = nrow(day))
      for (j in 1:length(var_range)){
        day[v] = var_range[j]
        preds = predict_via(day, data_use, prediction_scenario)
        output[j,] = preds
      }
      write.csv(output, paste0("~/PycharmProjects/physics_guided_nn/results_final/via/", prediction_scenario, "/preles_", data_use, "_", v, "_via_cond_", day_names[n], ".csv"))
      n = n+1
    }
  }
}

via("full", "temporal", gridsize = 200)
via("sparse", "temporal", gridsize = 200)

via("full", "spatial", gridsize = 200)
via("sparse", "spatial", gridsize = 200)

#out = PRELES(PAR = day$PAR, TAir = day$Tair, VPD = day$VPD, Precip = day$Precip, CO2 = day$CO2, fAPAR = day$fapar, p=CVfit[,4])$GPP
#plot(out, type='l')
#matplot(output, type='l')
