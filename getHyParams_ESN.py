# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import skopt
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel
import time
from joblib import Parallel, delayed, parallel_backend
from scipy.ndimage import median_filter
from skopt.plots import plot_convergence
from reservoirpy.nodes import Reservoir, Ridge
import reservoirpy as rpy
import h5py
import pickle
from tqdm import tqdm

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

# ========================================================================================================
# Setup Parameters
# ========================================================================================================
N_x = 64
FVMdataDirectory = "Data/FVM_5_topHat/"
variable = "Uf" # Choose between "Uf", RUcf

# Network parameters
N_units   = 256 # Number of neurons in reservoir 
leakyRate = 0.4 # Leaky rate of the reservoir
tikh_list = [1e-6, 1e-9] # Tikhonov factor (optimize among the values in this list)

# Validation parameters
N_val                = 5000
ensemble             = 14      # number of realizations to be optimized
printTimeDetailsFlag = False   # print details at each function evaluation
valFuncEvalPrintFlag = False

# Bayesian Optimization parameters
n_in = 0                                   # Number of Initial random points
rhoRange        = (0.1, 1)                 # range for spectral radius
sigmaInRangeLog = (-3, 1)                  # range for input scaling
nGridSearchX, nGridSearchY = 5, 5          # Number of points in the grid search
nBO = 10                                   # Number of points to be acquired through BO after grid search
nTot = nGridSearchX * nGridSearchY + nBO   # Total Number of Function Evaluatuions

# Save settings
saveModelInfoDirectory = "Data/ESN/modelInfo/"
saveModelDirectory     = "Data/ESN/model/"
saveBOoptDirectory     = "Data/ESN/BOopt/"

#========================================================================================================
# Functions & Classes
#========================================================================================================    
class postProcess_BOoptESN():
    """
    Class to reconstruct the Gaussian Process and plot the results.
    """
    def __init__(self, space, GPs, x_iters, f_iters, minimum):
        self.space = space
        self.plotResolution = 100  # Number of points to plot the GP reconstruction
        self.GPs = GPs
        self.f_iters = f_iters
        self.x_iters = x_iters
        self.minimum = minimum
        self._reconstructGP()

    def _reconstructGP(self):
        """Reconstruct the Gaussian Process for the given hyperparameters."""
        self.plotGridX, self.plotGridY = np.meshgrid(np.linspace(rhoRange[0],        rhoRange[1],        self.plotResolution), 
                                                     np.linspace(sigmaInRangeLog[0], sigmaInRangeLog[1], self.plotResolution))
        flattenCoords = np.column_stack((self.plotGridX.flatten(), self.plotGridY.flatten()))
        self.xGP      = self.space.transform(flattenCoords.tolist())     # GP prediction needs this normalized format 
        self.yGP      = np.zeros((ensemble, self.plotResolution, self.plotResolution))
        self.sGP      = np.zeros_like(self.yGP)

        for i in range(ensemble):
            mu, std = self.GPs[i].predict(self.xGP, return_std=True)

            # predGP = GP.predict(self.xGP)
            # amin = np.amin([10,self.f_iters.max()])
            # amax = -self.f_iters.min()
            # self.yGP[i] = np.clip(-predGP, a_min=-amin, a_max=amax).reshape(self.plotResolution,self.plotResolution) 
            self.yGP[i] = (-mu).reshape(self.plotResolution, self.plotResolution)
            self.sGP[i] = std.reshape(self.plotResolution, self.plotResolution)
                        # Final GP reconstruction for each realization at the evaluation points

        self.MeanGP = np.mean(self.yGP, axis=0) # Mean GP reconstruction
        self.stdGP = np.std(self.yGP, axis=0)
        self.UQGP  = np.sqrt(np.mean(self.sGP**2, axis=0))  # average predictive uncertainty


    def exportGPs(self):
        """Export the GP reconstruction."""
        return self.plotGridX, self.plotGridY, self.yGP, self.MeanGP, self.stdGP
    
    def plotAllGP(self, nPLotsX, nPLotsY):
        """
        Plot the GP reconstructions for all realizations.
        Args:
            nPLotsX: Number of plots in the x direction.
            nPLotsY: Number of plots in the y direction.
        """
        fig, axs = plt.subplots(nPLotsY, nPLotsX, figsize=(12, 8))

        for i in range(ensemble):
            ax = axs[i//nPLotsX, i%nPLotsX]

            tempCoontour = ax.contourf(self.plotGridX, self.plotGridY, self.yGP[i],
                                       levels=10, cmap='Blues')
            cbar = plt.colorbar(tempCoontour, ax=ax)
            cbar.set_label('-$\log_{10}$(MSE)',labelpad=15)
            ax.contour(self.plotGridX, self.plotGridY, self.yGP[i],
                       levels=10, colors='black', linewidths=0.5, linestyles='solid')  
            ax.scatter(self.x_iters[i, :nGridSearchX*nGridSearchY, 0],
                       self.x_iters[i, :nGridSearchX*nGridSearchY, 1],
                       c='r', marker='^')    # Grid Points
            ax.scatter(self.x_iters[i, nGridSearchX*nGridSearchY:, 0],
                       self.x_iters[i, nGridSearchX*nGridSearchY:, 1], 
                       c='lime', marker='o') # BO Points
            # ax.set_xlabel('Spectral Radius')
            # ax.set_ylabel('$\log_{10}$Input Scaling')
            ax.set_title('Realization \#'+ str(i+1))
        fig.supxlabel('Spectral Radius')
        fig.supylabel('$\log_{10}$Input Scaling')
        fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.5, rect=[0.05, 0.05, 0.95, 0.95])
        plt.show(block=False)   # Show without stopping the script
        plt.pause(0.5)          # Pause briefly so it renders (~0.5 sec)
        fig.savefig("AllGP_ensemble" + variable + "Ns" + str(nGridSearchX).zfill(2) + "Nx" + str(N_x).zfill(4) + "Nr" + str(N_units).zfill(4) + "LR" + str(int(leakyRate*10)).zfill(2) +".pdf", dpi=300, bbox_inches='tight', transparent=True)

    def plotMeanGP(self):
        """
        Plot the mean GP reconstruction.
        """
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        tempContour = axs[0].contourf(self.plotGridX, self.plotGridY, self.MeanGP,
                                     levels=10, cmap='Blues')
        axs[0].contour(self.plotGridX, self.plotGridY, self.MeanGP,
                       levels=10, colors='black', linewidths=0.5, linestyles='solid')
        cbar = plt.colorbar(tempContour, ax=axs[0])
        cbar.set_label('-$\log_{10}$(MSE)', labelpad=15)
        axs[0].scatter(self.minimum[:, 0], self.minimum[:, 1], 
                       c='lime', marker='^', edgecolors='k', linewidths=0.1)
        axs[0].set_title('Mean GP of the ensembles')
        axs[0].set_xlabel('Spectral Radius')
        axs[0].set_ylabel('$\log_{10}$Input Scaling')

        tempContour = axs[1].contourf(self.plotGridX, self.plotGridY, self.stdGP,
                                     levels=10, cmap='Reds')
        axs[1].contour(self.plotGridX, self.plotGridY, self.stdGP,
                       levels=10, colors='black', linewidths=0.5, linestyles='solid')
        cbar = plt.colorbar(tempContour, ax=axs[1])
        cbar.set_label('Standard Deviation', labelpad=15)
        axs[1].set_title('Standard Deviation of GP of the ensembles')
        axs[1].set_xlabel('Spectral Radius')
        axs[1].set_ylabel('$\log_{10}$Input Scaling')

        tempContour = axs[2].contourf(self.plotGridX, self.plotGridY, self.UQGP,
                                     levels=10, cmap='Greens')
        axs[2].contour(self.plotGridX, self.plotGridY, self.UQGP,
                       levels=10, colors='black', linewidths=0.5, linestyles='solid')
        cbar = plt.colorbar(tempContour, ax=axs[2])
        cbar.set_label('UQ', labelpad=15)
        axs[2].set_title('UQ of GP of the ensembles')
        axs[2].set_xlabel('Spectral Radius')
        axs[2].set_ylabel('$\log_{10}$Input Scaling')

        plt.tight_layout(pad=0.2, h_pad=0.5, w_pad=0.5, rect=[0.05, 0.05, 0.95, 0.95])
        plt.show(block=False)
        plt.pause(0.5)          # Pause briefly so it renders (~0.5 sec)
        fig.savefig("MeanGP_ensemble" + variable + "Ns" + str(nGridSearchX).zfill(2) + "Nx" + str(N_x).zfill(4) + "Nr" + str(N_units).zfill(4) + "LR" + str(int(leakyRate*10)).zfill(2) +".pdf", dpi=300, bbox_inches='tight', transparent=True)

#========================================================================================================
# Import Data
#========================================================================================================
with h5py.File(FVMdataDirectory + str(N_x).zfill(4) + "/primal.h5", "r") as h5:
    rawU = h5[variable][:]
print("rawU Data shape:", rawU.shape)

if variable == "RUcf":
    if N_x == 512:
        rawU = median_filter(rawU, size=(5, 7))
    elif N_x == 1024:
        rawU = median_filter(rawU, size=(9, 11))

# rawU = rawU[:50000,:] # O:nly for Short Dataset
# print("Short Data shape", rawU.shape)

# meanU = np.mean(rawU, axis=0)
# stdU = np.std(rawU, axis=0)
# U = (rawU - meanU) / stdU
U = rawU

dimX = U.shape[1]
N_train = U.shape[0]

U_tv = U[:-1,:].copy()
Y_tv = U[1: ,:].copy()

#========================================================================================================
# Bayesian Optimization Setup
#========================================================================================================
x1 = [[ # Grid search for the hyperparameters
    rhoRange[0]  + i * (rhoRange[1]  - rhoRange[0])  / (nGridSearchX - 1),
    sigmaInRangeLog[0] + j * (sigmaInRangeLog[1] - sigmaInRangeLog[0]) / (nGridSearchY - 1)
] for i in range(nGridSearchX) for j in range(nGridSearchY)]

# Range for hyperparameters
searchSpace = [Real(rhoRange[0],        rhoRange[1],        name='spectralRadius'),
               Real(sigmaInRangeLog[0], sigmaInRangeLog[1], name='inputScaling')]

# ARD 5/2 Matern Kernel with sigma_f in front for the Gaussian Process
kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-1, 3e0))*\
                  Matern(length_scale=[0.2,0.2], nu=2.5, length_scale_bounds=(5e-2, 1e1)) 
# kernel = ConstantKernel(1.0, (1e-3, 1e3)) \
#        * Matern(length_scale=[0.3, 1.0], length_scale_bounds=(1e-2, 1e2), nu=2.5)

# Hyperparameter Optimization using Grid Search + Bayesian Optimization
def makeObjective(U_tv_local, Y_tv_local, seed_local):
    def recycleVal(x):
        rho = x[0]
        sigma_in = 10 ** x[1]

        bestMSE = np.inf
        bestTikh = tikh_list[0]

        for tikh in tikh_list:
            reservoir = Reservoir(units=N_units, sr=rho, input_scaling=sigma_in, lr=leakyRate, seed=seed_local)
            readout = Ridge(ridge=tikh)
            esn = reservoir >> readout
            esn = esn.fit(U_tv_local, Y_tv_local)

            MSE_list = []
            for i in range(N_fo):
                p = N_in + i * N_fw
                X = U_tv[p : p + N_val].copy()
                Y = Y_tv[p : p + N_val].copy()

                Y_pred = esn.run(X)
                MSE = np.log10(np.mean((Y - Y_pred) ** 2))
                MSE_list.append(MSE)

            meanMSE = np.mean(MSE_list)
            if meanMSE < bestMSE:
                bestMSE = meanMSE
                bestTikh = tikh

        return bestMSE, bestTikh

    def wrapped(x):
        mse, tikh = recycleVal(x)
        wrapped.tikh_history.append(tikh)
        return mse

    wrapped.tikh_history = []
    return wrapped

def g(val):
    #Gaussian Process reconstruction
    b_e = GPR(kernel               = kernel,
              normalize_y          = True,  # If true mean assumed to be equal to the average of the obj function data, otherwise =0
              n_restarts_optimizer = 10,     # Number of random starts to find the gaussian process hyperparameters
            #   noise                = 1e-10, # Only for numerical stability
              alpha                =1e-6,
              random_state         = 10)    # seed
    
    #Bayesian Optimization
    res = skopt.gp_minimize(val,                         # the function to minimize
                      searchSpace,                       # the bounds on each dimension of x
                      base_estimator       = b_e,        # GP kernel
                      acq_func             = "EI",       # the acquisition function
                      n_calls              = nTot,       # total number of evaluations of f
                      x0                   = x1,         # Initial grid search points to be evaluated at
                      n_random_starts      = n_in,       # the number of additional random initialization points
                      n_restarts_optimizer = 3,          # number of tries for each acquisition
                      random_state         = 10,         # seed
                           )   
    return res

# Quantities to be saved
par               = np.zeros((ensemble, 4))      # GP parameters
x_iters           = np.zeros((ensemble,nTot,2))  # coordinates in hp space where f has been evaluated
f_iters           = np.zeros((ensemble,nTot))    # values of f at those coordinates
minimum           = np.zeros((ensemble, 4))      # minima found per each member of the ensemble
optimizationTimes = np.zeros(ensemble)           # time taken for optimization
trainTimes        = np.zeros(ensemble)           # time taken for training
GPs               = [None] * ensemble            # save the final gp reconstruction for each network
Wouts             = [None] * ensemble            # save the output weight matrix for each network
bias              = [None] * ensemble            # save the bias for each network
trainMSEs         = np.zeros(ensemble)           # MSE on the test set for each network

#========================================================================================================
# Validation
#========================================================================================================
N_fw = 2500 #(N_train-N_val)//(N_fo-1)  # how many steps forward the validation interval is shifted (in this way they are evenly spaced)
N_fo = int((N_train - N_fw)/N_fw) # 39 #int(N_train/N_val)     # number of validation intervals     19 for short dataset 39 for Full dataset
print("Number of validation intervals =", N_fo)
N_in = 0                          # timesteps before the first validation interval (can't be 0 due to implementation)
# !!!!! Warning: We are using 19 intervals of 5000 samples each, and the intervals are shifted by 2500 samples.

def runOptimization(i):
    rpy.verbosity(0)
    print(f"Realization    : {i+1}")
    seed = i + 1
    obj = makeObjective(U_tv, Y_tv, seed)

    t0 = time.time()
    res = g(obj)
    opt_time = time.time() - t0

    rhoBest, sigmaInLogBest = res.x
    sigmaInBest = 10 ** sigmaInLogBest
    min_index = np.argmin(res.func_vals)
    tikh_selected = obj.tikh_history[min_index]

    reservoirFinal = Reservoir(units=N_units, sr=rhoBest, input_scaling=sigmaInBest, lr=leakyRate, seed=seed)
    readoutFinal = Ridge(ridge=tikh_selected)
    t1 = time.time()
    esnFinal = (reservoirFinal >> readoutFinal).fit(U_tv, Y_tv)
    trainTime = time.time() - t1

    Y_test   = esnFinal.run(U_tv)
    trainMSE = float(np.mean((Y_test - Y_tv) ** 2))

    # outputModelFileName = "model" + variable + "Nx" + str(N_x).zfill(3) + "Nr" + str(N_units).zfill(4) + "LR" + str(int(leakyRate*10)).zfill(2) +  "Real" + str(seed).zfill(2) + ".pkl"
    # with open(saveModelDirectory + outputModelFileName, "wb") as f:
    #     pickle.dump(esnFinal, f)

    gp = res.models[-1]
    spaceGP = res.space   
    
    # params = gp.kernel_.get_params()
    # key = sorted(params)
    # par_vals = np.array([params[key[2]], params[key[5]][0], params[key[5]][1], gp.noise_])

    out = {
        "i": i,
        "gp": gp,
        "x_iter": np.asarray(res.x_iters, dtype=float),
        "f_iter": np.asarray(res.func_vals, dtype=float),
        "minimum_result": np.array([res.x[0], res.x[1], tikh_selected, res.fun], dtype=float),
        "opt_time": float(opt_time),
        "trainTime": float(trainTime),
        "WoutBest": esnFinal.nodes[-1].params["Wout"].copy(),
        "biasBest": esnFinal.nodes[-1].params["bias"].copy(),
        "trainMSE": trainMSE,
        "spaceGP": spaceGP,
    }

    print(f"[{i+1}/{ensemble}] BO done. Best val MSE: {res.fun} | tikh: {tikh_selected} | Time: {opt_time:.1f}s")

    return out




results = None
print("Starting Parallel...")
with parallel_backend("loky"):
    results = Parallel(n_jobs=7, verbose=10, temp_folder=saveBOoptDirectory)(
        delayed(runOptimization)(i) for i in range(ensemble)
    )
print("Parallel finished.")


# Save results
# Pre-allocate
for i, out in enumerate(results):
    idx = out["i"]
    GPs[idx]               = out["gp"]
    x_iters[idx]           = out["x_iter"]
    f_iters[idx]           = out["f_iter"]
    minimum[idx]           = out["minimum_result"]
    optimizationTimes[idx] = out["opt_time"]
    trainTimes[idx]        = out["trainTime"]
    Wouts[idx]             = out["WoutBest"]
    bias[idx]              = out["biasBest"]
    trainMSEs[idx]         = out["trainMSE"]
    print(f"Data for Realization {idx+1} is extracted.")

bestModelIdx = int(np.argmin([m[-1] for m in minimum]))
modelsInfo2Save = []
for i in range(ensemble):
    modelsInfo2Save.append({
        "Wout": Wouts[i],
        "bias": bias[i],
        "rho": minimum[i][0],
        "input_scaling": 10 ** minimum[i][1],
        "tikh": minimum[i][2],
        "loss": minimum[i][3],
        "seed": i + 1,
        "leakyRate": leakyRate,
        "dimX": dimX,
        "N_units": N_units,
        "trainTime": trainTimes[i],
    })

spaces = [out["spaceGP"] for out in results]
pp = postProcess_BOoptESN(spaces[0], GPs, x_iters, f_iters, minimum)
plotGridX, plotGridY, yGP, MeanGP, stdGP = pp.exportGPs()

#+========================================================================================================
# Save the results
#=========================================================================================================
outputBOoptFileName = "BOopt" + variable + "Nx" + str(N_x).zfill(4) + "Nr" + str(N_units).zfill(4) + "LR" + str(int(leakyRate*10)).zfill(2) + ".npz"
# outputBOoptFileName = "BOopt" + variable + "Ns" + str(nGridSearchX).zfill(2) +  "Nx" + str(N_x).zfill(4) + "Nr" + str(N_units).zfill(4) + "LR" + str(int(leakyRate*10)).zfill(2) + ".npz"
np.savez(saveBOoptDirectory + outputBOoptFileName,
         plotGridX = plotGridX, plotGridY = plotGridY,
         yGP = yGP, MeanGP = MeanGP, stdGP = stdGP,
         x_iters = x_iters, f_iters = f_iters,
         minimum = minimum, optimizationTime = optimizationTimes)


outputModelInfoFileName = "modelInfo" + variable + "Nx" + str(N_x).zfill(4) + "Nr" + str(N_units).zfill(4) + "LR" + str(int(leakyRate*10)).zfill(2) + ".npz"
# outputModelInfoFileName = "modelInfo" + variable + "Ns" + str(nGridSearchX).zfill(2) + "Nx" + str(N_x).zfill(4) + "Nr" + str(N_units).zfill(4) + "LR" + str(int(leakyRate*10)).zfill(2) + ".npz"
np.savez(saveModelInfoDirectory + outputModelInfoFileName,
         modelInfo=modelsInfo2Save,
         bestIdx=bestModelIdx)

#========================================================================================================
# Post-processing
#========================================================================================================
try:
    # pp.plotAllGP(4, 4)
    pp.plotMeanGP()
except Exception as e:
    print("Plotting failed:", e)


