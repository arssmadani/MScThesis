import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import h5py
from scipy.ndimage import median_filter

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')

# === USER PARAMETERS ===
Nc = 64
Nf = 2 * Nc
energyTH = 99
print("Computing for Coarse =", Nc, ", Fine =", Nf)
saveDirectory  = "Data/POD/"
variable = "Uf" # "RUcf" or "Uf"
FVMdataDirectory = "Data/FVM_5_topHat/"

#========================================================================================================
# Functions & Classes
#========================================================================================================    
def get_POD(U_init, rank):
    U_init = U_init.T  # Shape: (spatial, time)
    N_x = U_init.shape[0]
    N_t = U_init.shape[1]

    U_mean = np.mean(U_init, axis=1).reshape(N_x, 1)
    U = U_init - U_mean

    L, s, R = np.linalg.svd(U, full_matrices=False)

    L_M = L[:, :rank]
    S_M = np.diag(s[:rank])
    R_M = R[:rank, :]

    print("Shape of L:", L_M.shape, "S:",S_M.shape, "R:", R_M.shape)

    t1 = time.time()
    U_fluct = np.dot(L_M, np.dot(S_M, R_M))
    U_recon = U_fluct + U_mean
    reconstTime = time.time() - t1

    return U_recon.T, s, reconstTime  # Return shape: (time, space)


#========================================================================================================
# Load Data & Pre-Processing
#========================================================================================================
with h5py.File(FVMdataDirectory + str(Nc).zfill(4) + "/primal.h5", "r") as h5:
    U = h5[variable][:]

if variable == "RUcf":
    if Nc == 512:
        U = median_filter(U, size=(5, 7))
    elif Nc == 1024:
        U = median_filter(U, size=(9, 11))

#========================================================================================================
# Get POD Rank
#========================================================================================================
t0 = time.time()
_, eigenvalues, _ = get_POD(U, rank=U.shape[1])
prepTime = time.time() - t0
print("POD preparation time:", prepTime)
energyContribution = eigenvalues / np.sum(eigenvalues)
cumulativeEnergy = np.cumsum(energyContribution)

rank = np.searchsorted(cumulativeEnergy, energyTH/100) + 1
print("Number of modes to capture", energyTH, " % energy:", rank)

#========================================================================================================
# POD Reconstruction
#========================================================================================================
U_POD, _, timeReconstruct = get_POD(U, rank)

#========================================================================================================
# Performance Metrics
#========================================================================================================
# === POD COMPRESSION RATIO ===
CR_POD = U.size / (U.shape[1] + U.shape[1] * rank + rank * rank + rank * U.shape[0])
print('POD compression ratio:', CR_POD)
print('POD computational time:', timeReconstruct)

# === MSE ===
MSE = np.mean((U - U_POD) ** 2)
print('Reconstruction MSE:', MSE)

#========================================================================================================
# Plot
#========================================================================================================
# plt.figure(figsize=(10, 6))
# plt.plot(np.arange(1, U.shape[1] + 1), 100 * cumulativeEnergy, '-ok', markersize=5)
# plt.grid()
# plt.xlabel(r'Number of Modes Retained')
# plt.ylabel(r'Cumulative Energy Contribution [\%]')
# plt.xlim([0, U.shape[1]])
# plt.ylim([0, 105])
# plt.show()

#========================================================================================================
# Save Data
#========================================================================================================
# outputFileName = "reconstructed" + variable + "E" + str(energyTH).zfill(2) + "Nx" + str(Nc).zfill(4) + ".npz"
# np.savez(saveDirectory + outputFileName,
#          U_POD = U_POD, cumulativeEnergy = cumulativeEnergy, rank=rank,
#          CR_POD = CR_POD, MSE = MSE, timeReconstruct = timeReconstruct)

outputFileName = "prepTime" + variable + "E" + str(energyTH).zfill(2) + "Nx" + str(Nc).zfill(4) + ".npz"
np.savez(saveDirectory + outputFileName,
         prepTime = prepTime, timeReconstruct=timeReconstruct)

# if variable == "Uf":
#     with h5py.File(saveDirectory + "reconstructedUf4openFOAM/" + str(Nc).zfill(4) + "/reconstUf.h5", "w") as h5:
#         h5.create_dataset("Uf", data=U_POD)   
