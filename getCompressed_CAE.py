import torch
import torch.nn as nn
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
import time
import h5py
from scipy.ndimage import median_filter
import h5py

# Computation device
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')  # Apple Metal Performance Shaders (MPS)
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

print("Using device:", DEVICE)

# Visualization Settings
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

# Import Data Parameters
N_x    = 512
latentDim = 17
baseChannels = 4
maxChannels = N_x // 2
dt     = 1e-4
scheme = "cubic" # "linear", "cubic"
FVMdataDirectory = "Data/FVM_5_topHat/"
variable = "Uf" # Choose between "Uf", "RUcf"
modelDirectory  = "Data/CAE/model/"
paramsDirectory  = "Data/CAE/parameters/"   # where you saved stats

# Save settings
saveDirectory = "Data/CAE/reconstructedData/"


#========================================================================================================
# Functions & Classes
#========================================================================================================
def countParameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

class handleData_CAE:
    """
    Handles a single dataset for 1D CAE training.
    - Train-only normalization (z-score)
    - Gaussian noise added only to TRAIN set
    - Random split by default
    - Returns torch DataLoaders in shape (N, 1, D)
    """

    def __init__(self, U, DEVICE, addNoiseCount=0, noiseLevel=1e-4):
        assert isinstance(U, np.ndarray) and U.ndim == 2, "U must be (N_t, D) numpy array"
        self.U = U
        self.DEVICE = DEVICE
        self.addNoiseCount = int(addNoiseCount)
        self.noiseLevel = float(noiseLevel)
        self.dim = U.shape[1]

        self.meanU_ = None
        self.stdU_ = None

    @staticmethod
    def _add_noise_copies(X, copies, sigma):
        """Return X with `copies` noisy versions appended (noise from clean X each time)."""
        if copies <= 0:
            return X
        noisy = [X + np.random.normal(0.0, sigma, X.shape) for _ in range(copies)]
        return np.concatenate([X] + noisy, axis=0)

    def splitData(self, batchSize, subsampleRate=0, trainDataRatio=0.8, validDataRatio=0.2,
                  seed=42, num_workers=10):
        """
        Create train/valid DataLoaders.
        - Normalization stats computed on TRAIN set only
        - Noise added only to TRAIN set
        - Random split
        """
        assert abs(trainDataRatio + validDataRatio - 1.0) < 1e-9, "Ratios must sum to 1."

        # 1) optional subsample
        X = self.U[::subsampleRate, :] if subsampleRate and subsampleRate > 0 else self.U
        N = X.shape[0]

        # 2) random shuffle indices
        rng = np.random.default_rng(seed)
        order = rng.permutation(N)
        n_valid = int(validDataRatio * N)
        valid_idx = order[:n_valid]
        train_idx = order[n_valid:]

        # 3) train-only normalization
        meanU = X[train_idx].mean(axis=0)
        stdU  = X[train_idx].std(axis=0)
        stdU[stdU == 0.0] = 1.0
        self.meanU_, self.stdU_ = meanU, stdU

        Xn = (X - meanU) / stdU
        Xn_train = Xn[train_idx]
        Xn_valid = Xn[valid_idx]

        # 4) add noise to TRAIN only
        Xn_train = self._add_noise_copies(Xn_train, self.addNoiseCount, self.noiseLevel)

        # 5) convert to torch tensors in (N, 1, D)
        dataTrain = torch.tensor(Xn_train[None, ...], dtype=torch.float32).permute(1, 0, 2)
        dataValid = torch.tensor(Xn_valid[None, ...], dtype=torch.float32).permute(1, 0, 2)

        datasetTrain = TensorDataset(dataTrain, dataTrain)
        datasetValid = TensorDataset(dataValid, dataValid)

        train_loader = DataLoader(datasetTrain, batch_size=batchSize, shuffle=True,
                                  num_workers=num_workers, pin_memory=True,
                                  persistent_workers=(num_workers > 0))
        valid_loader = DataLoader(datasetValid, batch_size=batchSize, shuffle=False,
                                  num_workers=num_workers, pin_memory=True,
                                  persistent_workers=(num_workers > 0))
        return train_loader, valid_loader

    def getTestInputData(self):
        """
        Get normalized dataset as (N, 1, D) torch.Tensor on DEVICE.
        Uses train-fitted mean/std if available, else global stats.
        """
        X = self.U
        if self.meanU_ is None or self.stdU_ is None:
            meanU = X.mean(axis=0)
            stdU = X.std(axis=0)
            stdU[stdU == 0.0] = 1.0
        else:
            meanU, stdU = self.meanU_, self.stdU_

        Xn = (X - meanU) / stdU
        dataTest = torch.tensor(Xn[None, ...], dtype=torch.float32).permute(1, 0, 2)
        return dataTest.to(self.DEVICE, non_blocking=True)

    def denormalize(self, Xn):
        """
        Map normalized data back to original scale.
        Accepts np.ndarray or torch.Tensor.
        """
        if isinstance(Xn, torch.Tensor):
            Xn = Xn.detach().cpu().numpy()
        if Xn.ndim == 3 and Xn.shape[1] == 1:
            Xn = Xn[:, 0, :]  # remove channel dim

        meanU = self.meanU_
        stdU  = self.stdU_
        if meanU is None or stdU is None:
            meanU = self.U.mean(axis=0)
            stdU = self.U.std(axis=0)
            stdU[stdU == 0.0] = 1.0
        return Xn * stdU + meanU


class Autoencoder1D(nn.Module):
    """
    - Encoder: strided Conv1d blocks down to length=1.
    - Bottleneck: Linear stack to `latent_dim` (progressive halving), mirrored back.
    - Decoder: nearest-neighbor Upsample + Conv1d (no ConvTranspose1d).
    - Linear output (no activation) -> suitable for signed targets.
    """

    def __init__(
        self,
        seq_len: int,
        latent_dim: int,
        in_channels: int = 1,
        base_channels: int = 8,
        max_channels: int = 256,
        kernel_size: int = 4,
        use_bn: bool = True,
        negative_slope: float = 0.1,
    ):
        super().__init__()

        # ---- sanity checks -------------------------------------------------- #
        if seq_len & (seq_len - 1):
            raise ValueError("`seq_len` must be a power of two.")
        if kernel_size % 2 != 0:
            raise ValueError("`kernel_size` must be even.")
        if latent_dim < 1:
            raise ValueError("`latent_dim` must be a positive integer.")

        stride = 2
        pad = kernel_size // 2 - 1          # so length halves for stride=2
        Act = lambda ch: nn.LeakyReLU(negative_slope, inplace=True)

        # ---- CONV ENCODER --------------------------------------------------- #
        enc_layers = []
        self._enc_channels = [in_channels]   # keep for decoder mirroring
        self._enc_lengths  = [seq_len]

        length = seq_len
        in_ch = in_channels
        out_ch = base_channels

        while length > 1:
            # Conv block: Conv1d -> (BN) -> Act
            conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=not use_bn)
            enc_layers.append(conv)
            if use_bn:
                enc_layers.append(nn.BatchNorm1d(out_ch))
            enc_layers.append(Act(out_ch))

            # update trackers
            length = (length + 2 * pad - kernel_size) // stride + 1
            in_ch = out_ch
            out_ch = min(out_ch * 2, max_channels)
            self._enc_channels.append(in_ch)
            self._enc_lengths.append(length)

        # length should be 1 here
        self.encoder_conv = nn.Sequential(*enc_layers)
        self._conv_out_len = length          # == 1
        self._conv_out_ch  = in_ch
        flat_dim = self._conv_out_ch * self._conv_out_len  # == self._conv_out_ch

        # ---- FULLY-CONNECTED ENCODER (progressive halving) ------------------ #
        fc_enc_layers = []
        in_feat = flat_dim
        while in_feat > latent_dim:
            out_feat = max(latent_dim, in_feat // 2)
            fc_enc_layers.append(nn.Linear(in_feat, out_feat))
            if out_feat != latent_dim:
                fc_enc_layers.append(nn.LeakyReLU(negative_slope, inplace=True))
            in_feat = out_feat
        # self.encoder_fc = nn.Sequential(*fc_enc_layers)
        self.encoder_fc = nn.Sequential(nn.Linear(flat_dim, latent_dim))

        # ---- FULLY-CONNECTED DECODER (mirror) ------------------------------- #
        fc_dec_layers = []
        # collect encoder linear sizes to mirror
        enc_sizes = [m.out_features for m in self.encoder_fc if isinstance(m, nn.Linear)]
        dec_sizes = list(reversed(enc_sizes)) + [flat_dim] if enc_sizes else [flat_dim]

        in_feat = latent_dim
        for out_feat in dec_sizes:
            fc_dec_layers.append(nn.Linear(in_feat, out_feat))
            if out_feat != flat_dim:
                fc_dec_layers.append(nn.LeakyReLU(negative_slope, inplace=True))
            in_feat = out_feat
        # self.decoder_fc = nn.Sequential(*fc_dec_layers)
        self.decoder_fc = nn.Sequential(nn.Linear(latent_dim, flat_dim))

        # ---- CONV DECODER (upsample + conv; mirror of encoder_conv) --------- #
        # We will reverse the channel list and rebuild lengths by doubling.
        # Example: enc channels [1, 8, 16, 32, 64]  (lengths [512,256,128,64,32,1])
        # decoder needs: from 64 @ len=1 up to 1 @ len=512, doubling each step.
        rev_channels = list(reversed(self._enc_channels))   # e.g., [64, 32, 16, 8, 1]
        dec_layers = []
        cur_len = self._conv_out_len  # 1

        for idx in range(len(rev_channels) - 1):
            in_ch = rev_channels[idx]
            out_ch = rev_channels[idx + 1]

            # Upsample length x2
            dec_layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            cur_len *= 2

            # Conv to mix features after upsample (preserve length)
            conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn)
            dec_layers.append(conv)
            if idx != len(rev_channels) - 2:   # no BN/Act on the very last mapping to out_ch=in_channels
                if use_bn:
                    dec_layers.append(nn.BatchNorm1d(out_ch))
                dec_layers.append(nn.LeakyReLU(negative_slope, inplace=True))

        # If rounding caused a tiny mismatch, we’ll fix it in forward with pad/crop.
        self.decoder_conv = nn.Sequential(*dec_layers)

        # utility
        self.flatten = nn.Flatten()

        # ---- init ----------------------------------------------------------- #
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------------ #
    def forward(self, x):
        # Encoder
        x = self.encoder_conv(x)                           # (B, Cb, 1)
        x = self.flatten(x)                                # (B, Cb)
        z = self.encoder_fc(x)                             # (B, latent_dim)

        # Decoder
        x = self.decoder_fc(z)                             # (B, Cb)
        x = x.view(x.size(0), self._conv_out_ch, self._conv_out_len)  # (B, Cb, 1)
        x = self.decoder_conv(x)                           # (B, in_channels, ~seq_len)

        # Guard against off-by-one due to padding/rounding
        want_len = self._enc_lengths[0]                    # original seq_len
        got_len = x.size(-1)
        if got_len < want_len:
            # right-pad
            x = nn.functional.pad(x, (0, want_len - got_len))
        elif got_len > want_len:
            # center-crop the extra
            x = x[..., :want_len]

        return x


#========================================================================================================
# Import Data
#========================================================================================================
with h5py.File(FVMdataDirectory + str(N_x).zfill(4) + "/primal.h5", "r") as h5:
    U = h5[variable][:]
# U = U[10000:20000,:]
dim, N_t = U.shape[1], U.shape[0]

handler = handleData_CAE(U, DEVICE=DEVICE, addNoiseCount=0, noiseLevel=0)

# Load train-time stats (save these at the end of training!)
modelFileName = f"model{variable}Nx{str(N_x).zfill(4)}Latent{str(latentDim).zfill(2)}"
statsPath = paramsDirectory + modelFileName + ".npz"
try:
    stats = np.load(statsPath)
    # expected keys you should save during training:
    #   meanU, stdU  (vectors of length dim)
    handler.meanU_ = stats["meanU"]
    handler.stdU_  = stats["stdU"]
except Exception:
    # Fallback: use global stats (will still work, but not ideal)
    handler.meanU_ = U.mean(axis=0)
    std = U.std(axis=0); std[std == 0.0] = 1.0
    handler.stdU_ = std

dataTest = handler.getTestInputData()  # normalized using handler.mean/std


#========================================================================================================
# NN Setup
#========================================================================================================
model = Autoencoder1D(seq_len=dim, latent_dim=latentDim, max_channels=maxChannels, base_channels=baseChannels).to(DEVICE)
model.load_state_dict(torch.load(modelDirectory + modelFileName + '.pth'))
model.eval()

encoder_params = countParameters(model.encoder_conv) + countParameters(model.encoder_fc)
decoder_params = countParameters(model.decoder_fc) + countParameters(model.decoder_conv)
print(f"Encoder parameters: {encoder_params}")
print(f"Decoder parameters: {decoder_params}")

#=========================================================================================================
# Reconstruction
#=========================================================================================================
t0 = time.time()
with torch.no_grad():
    U_hat = model(dataTest)             # (N_t, 1, dim), normalized
t_recon = time.time() - t0
print("Reconstruction time:", t_recon)

# Denormalize using the handler (consistent with training stats)
U_hat = handler.denormalize(U_hat)      # returns (N_t, dim) in original scale

MSE = np.mean((U_hat - U) ** 2)
print('\nMSE:', MSE)

# Compression ratio (your original formula)
CR = U.size / (decoder_params + latentDim * U.shape[0] + 2)
print("CR:", CR)

#=========================================================================================================
# Save Data
#=========================================================================================================
outName = f"reconstructed{variable}Nx{str(N_x).zfill(4)}Latent{str(latentDim).zfill(2)}"

np.savez(saveDirectory + outName + ".npz",
        U_CAE=U_hat, MSE=MSE, CR=CR, timeReconstruct=t_recon,
        encoderParams=encoder_params, decoderParams=decoder_params)

if variable == "Uf":
    with h5py.File(saveDirectory + "/reconstructedUf4openFOAM/" + str(N_x).zfill(4) + "/reconstUf.h5", "w") as h5:
        h5.create_dataset("Uf", data=U_hat)                          # (nt, nz)