options(repos = c(CRAN = "https://cloud.r-project.org"))

# set install.packages lib path to user specific library path on HPC, if necessary!
lib_path = "/perm/pamw/R/library"
.libPaths(c(lib_path, .libPaths()))

if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv", lib = lib_path)
}

renv::restore()

source("PRELES_predictions.R")
source("via_conditional_preles.R")
source("std_effect.R")
