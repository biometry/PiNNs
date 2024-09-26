# Install devtools if not already installed
if (!require("devtools")) {
  install.packages("devtools")
  library(devtools)
}

# List of CRAN packages
cran_packages <- c("this.path", "BayesianTools")

github_packages <- c('MikkoPeltoniemi/Rpreles')

# Function to install missing CRAN packages
install_cran_packages <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE)) {
      install.packages(pkg, dependencies = TRUE)
      library(pkg, character.only = TRUE)
    }
  }
}

# Function to install PRELES, if missing
install_github_packages <- function(packages) {
  for (pkg in packages) {
    pkg_name <- strsplit(pkg, "/")[[1]][2]  # Extract the package name
    if (!require(pkg_name, character.only = TRUE)) {
      devtools::install_github(pkg, ref="ecolmod-version")
      library(pkg_name, character.only = TRUE)
    }
  }
}

# Install CRAN and GitHub packages
install_cran_packages(cran_packages)
install_github_packages(github_packages)
