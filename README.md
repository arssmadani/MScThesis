# MScThesis

WARNING: The CAE codes are optimized to be used on a NVIDIA P40 Tesla GPU !!

The main functions of the codes are:

getCompressed_POD.py : Compress and reconstruct Primal Solutions or Injected Residuals using Proper Orthogonal Decomposition (POD) via Singular Value Decomposition (SVD).

getHyParams_ESN.py : Get best Echo State Network (ESN) hyperparameters using a Bayesian Optimization framework to compress and reconstruct Primal Solutions or Injected Residuals.

getCompressed_ESN.py : Compress the Primal Solutions or Injected Residuals using the best hyperparameters optimized from getHyParams_ESN.py

getModel_CAE.py : Training and validation phase of compression and reconstruction of Primal Solutions or Injected Residuals using Convolutional Autoencoder (CAE). 

getCompressed_CAE.py : Test phase of compression and reconstruction of Primal Solutions or Injected Residuals using Convolutional Autoencoder (CAE). 
