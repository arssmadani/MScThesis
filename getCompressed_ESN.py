import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigs as sparse_eigs
import time
import h5py
from scipy.ndimage import median_filter
from reservoirpy.nodes import Reservoir, Ridge
import reservoirpy as rpy
from joblib import Parallel, delayed, parallel_backend
import pickle
from functions import importData, handleData_ESN, ESN1D
from tqdm import tqdm

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

#===============================================================================
# Setup Parameters
#===============================================================================
N_x = 512
# nGridSearchX = 8
FVMdataDirectory = "Data/FVM_5_topHat/"
variable = "Uf" # Choose between "Uf", "RUcf"

modelDirectory     = "Data/ESN/model/"
modelInfoDirectory = "Data/ESN/modelInfo/"
saveDirectory      = "Data/ESN/reconstructedData/"

# Network parameters
N_units  = 256 # Number of neurons in reservoir
leakyRate = 0.4 # Leaky rate of the reservoir

ensemble = 14
printTimeDetailsFlag = False # print details at each function evaluation

N_intt    = 100    # Number of steps in each test interval
#========================================================================================================
# Import Data
#========================================================================================================
with h5py.File(FVMdataDirectory + str(N_x).zfill(4) + "/primal.h5", "r") as h5:
    rawU = h5[variable][:].astype(np.float32)

if variable == "RUcf":
    if N_x == 512:
        rawU = median_filter(rawU, size=(5, 7))
    elif N_x == 1024:
        rawU = median_filter(rawU, size=(9, 11))

print("Data shape:", rawU.shape)

# rawU = rawU[:50000,:] # Only for Short Dataset
# print("Short Data shape:", rawU.shape)

# meanU = np.mean(rawU, axis=0)
# stdU = np.std(rawU, axis=0)
# U = (rawU - meanU) / stdU
U = rawU.astype(np.float32)

# meanU = np.mean(rawU, axis=0).astype(np.float32)
# stdU = np.std(rawU, axis=0).astype(np.float32)
# U = ((rawU - meanU) / stdU).astype(np.float32)

dimX = U.shape[1]
N_total  = U.shape[0] 
N_train  = N_total - N_intt
#========================================================================================================
# Import ESN Models
#========================================================================================================
# modelInfoFileName = "modelInfo" + variable + "Ns" + str(nGridSearchX).zfill(2) + "Nx" + str(N_x).zfill(4) + "Nr" + str(N_units).zfill(4) + "LR" + str(int(leakyRate*10)).zfill(2) + ".npz"
modelInfoFileName = "modelInfo" + variable + "Nx" + str(N_x).zfill(4) + "Nr" + str(N_units).zfill(4) + "LR" + str(int(leakyRate*10)).zfill(2) + ".npz"
modelsInfo_RAW = np.load(modelInfoDirectory + modelInfoFileName, allow_pickle=True)

modelsInfo = modelsInfo_RAW["modelInfo"]
bestIdx = int(modelsInfo_RAW["bestIdx"])
bestModel = modelsInfo[bestIdx]  # extract dict

print("Number of Realizations", ensemble)

#========================================================================================================
# Data Reconstruction
#========================================================================================================
N_test = N_train // N_intt
print("Number of Test Intervals:", N_test)

def reconstruct(i):
    rpy.verbosity(0)
    print("Processing Realization", i + 1, "of", ensemble)
    U_ESN_TEMP = np.zeros((N_total, dimX), dtype=np.float32)
    U_ESN_TEMP[:N_intt, :] = U[:N_intt]

    seed = i + 1
    ModelInfo = modelsInfo[i]

    Wout = ModelInfo["Wout"].astype(np.float32)
    bias = ModelInfo["bias"].astype(np.float32)

    bare = Reservoir(units=N_units,
                    sr=ModelInfo["rho"],
                    lr=ModelInfo["leakyRate"],
                    input_scaling=ModelInfo["input_scaling"], 
                    seed=seed)

    # Loop through test intervals
    timeStart = time.time()
    for j in tqdm(range(N_test), desc="Processing Intervals", leave=False):
        idx_start = j * N_intt
        idx_end   = idx_start + N_intt
        if idx_end + N_intt > N_total:
            print("break on")
            break  # avoid incomplete block at end

        wash_in = U[idx_start : idx_end].copy()
        bare.run(wash_in, reset=True)  # reach stable internal state
        xa = bare.state().squeeze().astype(np.float32)

        # Closed-loop prediction
        pred = np.empty((N_intt, dimX), dtype=np.float32)
        for k in range(N_intt):
            y = xa @ Wout + bias
            pred[k] = y
            bare.run(y, reset=False)
            xa = bare.state().squeeze().astype(np.float32)

        # Store predictions
        pred_start = idx_end
        pred_end   = idx_end + N_intt
        U_ESN_TEMP[pred_start:pred_end, :] = pred

    # de-normalize
    # U_ESN_TEMP *= stdU
    # U_ESN_TEMP += meanU

    testTime_TEMP = time.time() - timeStart

    testMSE_TEMP = float(np.mean((U_ESN_TEMP - rawU) ** 2))
    compressRatio_TEMP = float(U.size / (Wout.size + bias.size + 5 + xa.size * (N_test + 1)))
    
    # ---- SAVE THIS ENSEMBLE TO DISK AND RETURN ONLY METRICS ----
    ens_file = os.path.join(
        saveDirectory,
        f"reconstructed_{variable}_ens{i+1:02d}_Nx{N_x:04d}Nr{N_units:04d}LR{int(leakyRate*10):02d}.npy"
    )
    np.save(ens_file, U_ESN_TEMP)   # .npy per-ensemble

    print("Ens", i+1, "| MSE:", testMSE_TEMP, "| CR:", compressRatio_TEMP, "| Time:", testTime_TEMP)
    return i, ens_file, testMSE_TEMP, compressRatio_TEMP, testTime_TEMP

with parallel_backend("loky"):
    results = Parallel(
        n_jobs=7,  # keep it small to fit RAM
        verbose=10,
        temp_folder=saveDirectory            # place memmaps/temp here
    )(delayed(reconstruct)(i) for i in range(ensemble))

# Save results
testMSE        = np.zeros(ensemble, dtype=np.float32)
compressRatio  = np.zeros(ensemble, dtype=np.float32)
testTime       = np.zeros(ensemble, dtype=np.float32)
ens_paths      = [None]*ensemble

for i, (idx, path, mse, cr, ttime) in enumerate(results):
    ens_paths[idx]  = path
    testMSE[idx]    = mse
    compressRatio[idx] = cr
    testTime[idx]   = ttime

#========================================================================================================
# Save Data
#========================================================================================================
outputFileName = f"reconstructedInfo{variable}Nx{N_x:04d}Nr{N_units:04d}LR{int(leakyRate*10):02d}.npz"
np.savez(os.path.join(saveDirectory, outputFileName),
         ensemble_files=np.array(ens_paths, dtype=object),
         testTime=testTime,
         compressRatio=compressRatio,
         testMSE=testMSE)
print("Saved metrics + file list:", outputFileName)

# creates a memmapped 3D array on disk, then fills it
mm_path = os.path.join(saveDirectory, f"U_ESN_mm_{N_x:04d}_{N_units:04d}.dat")
U_ESN_mm = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=(ensemble, N_total, dimX))
for i, p in enumerate(ens_paths):
    U_ESN_mm[i, :, :] = np.load(p, mmap_mode="r")
U_ESN_mm.flush()
print("Stitched memmap at:", mm_path)

# After stitching into U_ESN_mm:
# npz_path = os.path.join(saveDirectory, f"reconstructedData{variable}Ns{nGridSearchX:02d}Nx{N_x:04d}Nr{N_units:04d}LR{int(leakyRate*10):02d}.npz")
npz_path = os.path.join(saveDirectory, f"reconstructedData{variable}Nx{N_x:04d}Nr{N_units:04d}LR{int(leakyRate*10):02d}.npz")

# Option 1: If final array fits in RAM now:
U_ESN_full = np.array(U_ESN_mm)  # loads into memory
# del U_ESN_mm
np.savez(npz_path,
         U_ESN=U_ESN_full,
         testTime=testTime,
         compressRatio=compressRatio,
         testMSE=testMSE)
print("Saved combined .npz:", npz_path)

if variable == "Uf":
    U_ESN = np.mean(U_ESN_mm, axis=0)
    print(np.mean((U_ESN-rawU)**2))
    with h5py.File(saveDirectory + "reconstructedUf4openFOAM/" + str(N_x).zfill(4) + "/reconstUf.h5", "w") as h5:
        h5.create_dataset("Uf", data=U_ESN)  