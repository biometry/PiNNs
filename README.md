# init
This is the repo of the Physics-informed Neural Network project.<br/>

It's structured as follows:
- ./src: contains the source code of the model PRELES
- ./data: contains the data used to force PRELES and the neural networks (in the pre-processed version available on request)
- ./r: contains the R scripts used to calibrate PRELES
- ./misc: contains the scripts shared among prediction experiments
- ./spatial, ./temporal, ./spatio-temporal: contain the scripts used to run the prediction experiments

To run the code locally, create a 'results' subfolder in each prediction directory.
Data used to force PRELES and the neural networks can be found at: https://github.com/COST-FP1304-PROFOUND/ProfoundData<br/>
C source code of the model PRELES in ./src can be found at: https://github.com/MikkoPeltoniemi/Rpreles<br/>

