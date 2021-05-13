# Data preparation
# data available on https://github.com/florianhartig/ProfoundData

if (!require("ProfoundData")){
  devtools::install_github(repo = "COST-FP1304-PROFOUND/ProfoundData", 
                           subdir = "ProfoundData", 
                           dependencies = T, build_vignettes = T)}
library("ProfoundData")
?ProfoundData
# to project data folder
setwd("C:/Users/Niklas/Desktop/Uni/M.Sc. Environmental Science/Thesis/Official Project Folder/data/ProfoundData")
downloadDatabase()



browseData('ClimateData')
summarizeData()

