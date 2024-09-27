# Process-informed neural networks: a hybrid modelling approach to improve predictive performance and inference of neural networks in ecology and beyond

This repository contains the code that accompanies the article (DOI). The process-based model PRELES is combined with neural networks into process-informed neural networks (PINNs), which incorporate the process model knowledge into the neural network training algorithm and structure. Prediction tasks for C-fluxes in temperate forests are systematically evaluated with five different types of PINNs (i) in data-rich and data-sparse regimes (ii) in temporal, spatial and spatio-temporal prediction scenarios. <br/>

The repository is structured as follows:
- ./src: contains the source code of the model PRELES
- ./data: contains the data used to force PRELES and the neural networks (for the pre-processed version, see OSF repository: DOI 10.17605/OSF.IO/7GZBN)
- ./r: contains the R scripts used for PRELES calibration and variable importance analyses
- ./misc: contains the scripts shared among prediction experiments
- ./spatial, ./temporal, ./spatio-temporal: contain the scripts used to run the experiments at three different prediction scenarios

Data source for creating PRELES and the neural networks forcing can be found at: https://github.com/COST-FP1304-PROFOUND/ProfoundData<br/>
C source code of the model PRELES in ./src can be found at: https://github.com/MikkoPeltoniemi/Rpreles<br/>

## Set up computing environment


- R: The scripts run safely in R v4.3.1 / v4.2.1 with the package versions listed in requirements_r.txt. Packages can be installed with "install_packages.R", missing packages are installed on the fly when running main.R. 

- Python: Package versions and dependencies are listed in requirements.txt. Create a virtual environment before installtion, e.g. directly in the project directory with
```console
python -m venv pinns
source venv/bin/activate

pip install -r requirements.txt
```

- Preles-Compilation: For PRELES compilation, see README file in src folder.

## Installation of R and Python packages