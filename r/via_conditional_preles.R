#==========================#
# VIA PRELES / conditional #
#==========================#

library(Rpreles)

predict_via <- function(day, data_use, prediction_scenario, point_wise = FALSE){
  
  if (prediction_scenario =='exp1'){
    CVfit = read.csv(paste0("~/PycharmProjects/physics_guided_nn/data/Psinglesite_CVfit_", data_use, ".csv"))#[,2:6]
  }else{
    CVfit = read.csv(paste0("~/PycharmProjects/physics_guided_nn/data/Pmultisite_CVfit_", data_use, "_", prediction_scenario, ".csv"))[,2:5]
  }
  
  if (prediction_scenario == 'exp1'){
    preds = matrix(nrow = nrow(day), ncol=ncol(CVfit))
    for (i in 1:ncol(CVfit)){
      preds[,i] = PRELES(PAR=day$PAR, TAir=day$Tair, VPD=day$VPD, Precip=day$Precip, CO2=day$CO2, fAPAR=day$fapar, p=CVfit[,i])$GPP
    }
    preds = apply(preds, 1, mean)
  }else{
    if (point_wise){
      points <- unique(day$X)
      preds_sitewise <- numeric(length(points))
      for (point in points){
        df <- day[day$X==point,]
        preds <- matrix(nrow = nrow(df), ncol=ncol(CVfit))
        for (i in 1:ncol(CVfit)){
          preds[,i] = PRELES(PAR=df$PAR, TAir=df$Tair, VPD=df$VPD, Precip=df$Precip, CO2=df$CO2, fAPAR=df$fapar, DOY = df$DOY, p=CVfit[,i])$GPP
        }
        preds <- apply(preds, 1, mean)
        preds_sitewise[point] <- preds
      }
    }else{
      df = day
      preds <- matrix(nrow = nrow(df), ncol=ncol(CVfit))
      for (i in 1:ncol(CVfit)){
          preds[,i] = PRELES(PAR=df$PAR, TAir=df$Tair, VPD=df$VPD, Precip=df$Precip, CO2=df$CO2, fAPAR=df$fapar, DOY = df$DOY, p=CVfit[,i])$GPP
      }
      preds <- apply(preds, 1, mean)
      }
    }
  return(preds)
}


via <- function(data_use, prediction_scenario, gridsize = 200, point_wise = FALSE){
  
  if (prediction_scenario == 'exp1'){
    hyytiala <- read.csv(paste0("~/PycharmProjects/physics_guided_nn/data/hyytialaF_", data_use, ".csv"))
  }else{
    hyytiala <- read.csv(paste0("~/PycharmProjects/physics_guided_nn/data/allsitesF_", prediction_scenario, "_", data_use, ".csv"))
  }
  
  
  
  hyytiala$date <- as.Date(hyytiala$date)
  hyytiala$year <- format(hyytiala$date, format="%Y")
  
  if (prediction_scenario == 'exp1'){
    hyytiala_train <- hyytiala[!(hyytiala$year %in% c("2008", "2007", "2005", "2004")), ]

    hyytiala_test <- hyytiala[hyytiala$year == "2008", ]
    
  }else if (prediction_scenario == 'exp2'){
    
    hyytiala_train <- hyytiala[(hyytiala$site %in% c("sr","bz", "ly", "co")), ]
    hyytiala_test <- hyytiala[((hyytiala$site == "h") & (hyytiala$year == "2008")), ]
    
  }else if (prediction_scenario =='exp3'){
    
    hyytiala_train <- hyytiala[(hyytiala$site %in% c("sr","bz", "ly", "co")), ]
    hyytiala_train <- hyytiala_train[(hyytiala_train$year %in% c("2005", "2004")), ]
    hyytiala_test <- hyytiala[((hyytiala$site == "h") & (hyytiala$year == "2008")), ]
    
  }
    
  hyytiala_test <- hyytiala_test[2:nrow(hyytiala_test),] 
  
  variables = c('PAR', 'Tair', 'VPD', 'Precip', 'fapar')
  
  thresholds = data.frame('PAR' = c(min(hyytiala_test$PAR), max(hyytiala_test$PAR)), 'Tair'= c(min(hyytiala_test$Tair), max(hyytiala_test$Tair)), 'VPD' = c(min(hyytiala_test$VPD), max(hyytiala_test$VPD)), 'Precip'= c(min(hyytiala_test$Precip), max(hyytiala_test$Precip)), 'fapar'= c(min(hyytiala_test$fapar), max(hyytiala_test$fapar)))
  
  for (v in variables){
    
    var_range = seq(thresholds[v][1,], thresholds[v][2,], length.out=gridsize)
    day_names = c("mar", "jun", "sep", "dec")
    x_test = hyytiala_test
    
    if (prediction_scenario == 'spatial'){x_test$X = gsub("[^a-zA-Z]", "", x_test$X)}
    
    mar = with(x_test, x_test[(date >= "2008-03-13" & date <= "2008-03-27"),])[,c(3:14)]
    jun = with(x_test, x_test[(date >= "2008-06-14" & date <= "2008-06-28"),])[,c(3:14)]
    sep = with(x_test, x_test[(date >= "2008-09-13" & date <= "2008-09-27"),])[,c(3:14)]
    dec = with(x_test, x_test[(date >= "2008-12-14" & date <= "2008-12-28"),])[,c(3:14)]
  
    days = list(mar, jun, sep, dec)
    
    n = 1
    for (day in days){
      day = day
      output = matrix(nrow = length(var_range), ncol = nrow(day))
      for (j in 1:length(var_range)){
        day[v] = var_range[j]
        preds = predict_via(day, data_use, prediction_scenario, point_wise = point_wise)
        output[j,] = preds
      }
      write.csv(output, paste0("~/PycharmProjects/physics_guided_nn/results_", prediction_scenario, "/via/preles_", data_use, "_", v, "_via_cond_", day_names[n], ".csv"))
      n = n+1
    }
  }
}

via(data_use = "full",prediction_scenario =  "exp1", gridsize = 200)
via(data_use = "sparse",prediction_scenario =  "exp1", gridsize = 200)

via(data_use = "full",prediction_scenario =  "exp2", gridsize = 200, point_wise = FALSE)
via(data_use = "sparse",prediction_scenario =  "exp2", gridsize = 200, point_wise = FALSE)

via(data_use = "full", prediction_scenario = "exp3", gridsize = 200, point_wise = FALSE)
via(data_use = "sparse", prediction_scenario = "exp3", gridsize = 200, point_wise = FALSE)


PRELES()