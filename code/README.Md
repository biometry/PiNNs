# CODE

## Structure
Scripts are designed separately, one script per experiment and model type.

There are two types of experiments:
- temporal prediction
- spatial prediction (scripts include a "2" in the first digits)

Specific experiments:
- Overfitting ("of")
- Network architecture search ("EN")
- Finetuning of learning rate ("ft")
- Training \& evaluation ("eval")

Models:
- Naive Neural Network ("mlp")
- Bias Correction ("res")
- Residual Physics ("res2")
- Physical Regularisation ("reg")
- Physical Embedding ("emb")

Others:
- Training pipeline is defined in *training.py*
- Hyperparameter and Architecture Search is defined in *HP.py*
- Utility functions are defined in *utils.py*
- Data cleaning pipeline is defined in *dataset.py*
- Stochastic L-BFGS algorithm is defined in *slbfgs.py* according to Yatawatta *et al.* (2019)



S. Yatawatta, L. De Clercq, H. Spreeuw and F. Diblen, "A Stochastic LBFGS Algorithm for Radio Interferometric Calibration," 2019 IEEE Data Science Workshop (DSW), 2019, pp. 208-212, doi: 10.1109/DSW.2019.8755567
- github: [Link](https://github.com/SarodYatawatta/federated-pytorch-test) (algorithm in ./src/lbfgsnew.py)

