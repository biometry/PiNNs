# Run main.R

## Sets up your R-environment

After cloning into PiNNs, you followed the instructions on setting up the environment. One prequisite is an installation of the renv package. This is installed in the header of main, if not already installed. Afterwards, the environment is restored from renv by installing all required package versions and dependencies.

## Calibrates and predicts with PRELES

By calling main.R we source PRELES_predictions. In the current version, the flag save_data = FALSE. If you set this to TRUE, the current files allsites_ and hyytiala_ will be overwritten with your PRELEs calibration. Independent of the flag, this call will create a results folder in each prediction experiment. It will save the calibrated parameters to the data folder from which they will be loaded in the next step. 

## Conducts variable important analysis.

We also source via_conditional_preles. Loads the hyytiala_ and the allsites_ data, as well as calibrated PRELES parameters from the data folder. The output will be written to the respective results folder.
