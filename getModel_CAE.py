import torch
import torch.nn as nn
from torchsummary import summary
from torchview import draw_graph
from torchviz import make_dot
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
import time
from scipy.ndimage import median_filter
import h5py
from torch import amp
from tqdm import tqdm
from sklearn.decomposition import PCA
import h5py

# Computation device
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')  # Apple Metal Performance Shaders (MPS)
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

print("Using device:", DEVICE)

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

N_x    = 64
dt     = 1e-4
scheme = "cubic" # "linear", "cubic"
FVMdataDirectory = "Data/FVM_5_topHat/"
variable = "RUcf" # Choose between "Uf", "RUcf"

batchSize  = 1000
N_epochs   = 500
subsampleRate = 0 # R: 0  U: 5
baseChannels = 4  # R: 4  U: 4

printModelFlag = True # Print model summary

saveParamsDirectory = "Data/CAE/parameters/"
saveModelDirectory  = "Data/CAE/model/" 
saveModelTracerDirectory  = "Data/CAE/modelTracer/"

#========================================================================================================
# Functions & Classes
#========================================================================================================
class CAE():
    def __init__(self, printEpochsFlag = True):
        self.printEpochsFlag = printEpochsFlag

    def train(self, epoch):
        epoch_loss = 0
        # for batch in dataLoaderTrain:
        for batch in tqdm(dataLoaderTrain, desc="Training Intervals", leave=False):
            input, target = batch[0].to(DEVICE, dtype=torch.float, non_blocking=True), batch[1].to(DEVICE, dtype=torch.float, non_blocking=True)
            optimizer.zero_grad()
            with amp.autocast(device_type='cuda'):
                forward_pass = model(input)
                loss = criterion(forward_pass, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            # print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(dataLoaderTrain), loss.item()))
        if self.printEpochsFlag:
            print(f"Epoch {epoch} | LR: {optimizer.param_groups[0]['lr']}")
            print("===> Training Epoch {} Complete: Avg. Loss: {}".format(epoch, epoch_loss / len(dataLoaderTrain)))
        return epoch_loss / len(dataLoaderTrain)

    def validate(self, epoch):
        avg_psnr = 0
        avg_mse = 0
        with torch.no_grad():
            for batch in dataLoaderValid:
                input, target = batch[0].to(DEVICE, dtype=torch.float), batch[1].to(DEVICE, dtype=torch.float)
                prediction = model(input)
                mse = criterion(prediction, target)
                psnr = 10 * np.log10(1 / mse.item())
                avg_psnr += psnr
                avg_mse += mse.item()
        # print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(dataLoaderValid)))
        print("===> Validation Epoch {} Complete: Avg. Loss: {}".format(epoch, avg_mse / len(dataLoaderValid)))
        return avg_mse / len(dataLoaderValid)


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


class ModelTracer:
    """
    Utility to trace a PyTorch model and collect layer-by-layer details:
    - name, type, in/out shapes, params
    - layer-specific hyperparameters (Conv1d, Linear, BN, etc.)
    """

    def __init__(self, model: nn.Module, example_input: torch.Tensor):
        self.model = model.eval()  # eval mode avoids BatchNorm issues
        self.example_input = example_input
        self.records = []
        self._trace()

    def _count_params(self, m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    def _layer_hyperparams(self, mod: nn.Module):
        hp = {}
        if isinstance(mod, nn.Conv1d):
            hp |= {
                "in_ch": mod.in_channels,
                "out_ch": mod.out_channels,
                "kernel_size": mod.kernel_size,
                "stride": mod.stride,
                "padding": mod.padding,
                "dilation": mod.dilation,
                "groups": mod.groups,
                "bias": mod.bias is not None
            }
        elif isinstance(mod, nn.BatchNorm1d):
            hp |= {
                "num_features": mod.num_features,
                "eps": mod.eps,
                "momentum": mod.momentum,
                "affine": mod.affine,
                "track_stats": mod.track_running_stats
            }
        elif isinstance(mod, nn.Linear):
            hp |= {
                "in_features": mod.in_features,
                "out_features": mod.out_features,
                "bias": mod.bias is not None
            }
        elif isinstance(mod, nn.Upsample):
            hp |= {"mode": mod.mode, "scale_factor": mod.scale_factor}
        elif isinstance(mod, nn.LeakyReLU):
            hp |= {"negative_slope": mod.negative_slope, "inplace": mod.inplace}
        elif isinstance(mod, nn.Flatten):
            hp |= {"start_dim": mod.start_dim, "end_dim": mod.end_dim}
        return hp

    def _trace(self):
        handles = []

        def hook(mod, inp, out, name):
            in_shape = tuple(inp[0].shape) if isinstance(inp, (tuple, list)) else tuple(inp.shape)
            out_shape = tuple(out.shape) if not isinstance(out, (tuple, list)) else tuple(out[0].shape)
            rec = {
                "name": name,
                "type": mod.__class__.__name__,
                "in_shape": in_shape,
                "out_shape": out_shape,
                "params": self._count_params(mod),
            }
            rec |= self._layer_hyperparams(mod)
            self.records.append(rec)

        for name, module in self.model.named_modules():
            if not name or isinstance(module, nn.Sequential):
                continue
            handles.append(module.register_forward_hook(lambda mod, inp, out, n=name: hook(mod, inp, out, n)))

        with torch.no_grad():
            _ = self.model(self.example_input)

        for h in handles:
            h.remove()

        # Drop duplicates from multiple forward calls
        self.df = pd.DataFrame(self.records).drop_duplicates(subset=["name"]).reset_index(drop=True)

    def to_dataframe(self) -> pd.DataFrame:
        return self.df

    def to_csv(self, path: str):
        self.df.to_csv(path, index=False)

    def to_latex(self, path: str, caption="Layer details", label="tab:model_layers"):
        df_ltx = self.df.copy()
        df_ltx["in_shape"] = df_ltx["in_shape"].apply(lambda s: f"`{s}`")
        df_ltx["out_shape"] = df_ltx["out_shape"].apply(lambda s: f"`{s}`")
        df_ltx["name"] = df_ltx["name"].apply(lambda s: s.replace("_", r"\_"))
        latex = df_ltx.to_latex(
            index=False,
            escape=False,
            caption=caption,
            label=label
        )
        with open(path, "w") as f:
            f.write(latex)

    def summary(self):
        total_params = int(self.df["params"].sum())
        return f"Total trainable parameters: {total_params}\nLayers counted: {len(self.df)}"

def plotPerformance(trainLoss, validLoss, lr):
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

    axs[0].semilogy(range(1, N_epochs + 1), lr, '-^k', markersize=2.5)
    axs[0].set_ylabel('Learning Rate')
    axs[0].set_xlim(0, N_epochs)
    axs[0].grid()

    axs[1].semilogy(range(1, N_epochs + 1), trainLoss, '-^r', label=r'Training Data', markersize=2.5)
    axs[1].semilogy(range(1, N_epochs + 1), validLoss, '-^b', label=r'Validation Data', markersize=2.5)
    axs[1].legend()
    axs[1].set_ylabel('MSE')
    axs[1].set_xlabel('Number of Epochs')
    axs[1].grid()

    plt.tight_layout()
    plt.show()
    plt.savefig(saveParamsDirectory + "performance_" + variable + "Nx" + str(N_x).zfill(3) + "Latent" + str(latentDim).zfill(2) + ".pdf", bbox_inches='tight', dpi=300)
    plt.close

def countParameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

#========================================================================================================
# Import Data
#========================================================================================================
with h5py.File(FVMdataDirectory + str(N_x).zfill(4) + "/primal.h5", "r") as h5:
    U = h5[variable][:]

print("Data shape:", U.shape)
dim  = U.shape[1]

if variable == "RUcf":
    if N_x == 512:  
        U = median_filter(U, size=(5, 7))
    elif N_x == 1024:
        U = median_filter(U, size=(9, 11))

pca = PCA(n_components=0.99)  # keep 99% variance
pca.fit(U)                    # U: shape (time, features)
latentDim = pca.n_components_
print("Recommended latent dim:", latentDim)

dataHandler = handleData_CAE(U, DEVICE=DEVICE, addNoiseCount=0, noiseLevel=0)

dataLoaderTrain, dataLoaderValid = dataHandler.splitData(batchSize, subsampleRate, trainDataRatio=0.8, validDataRatio=0.2)
print("Data Loaders Are Created!")

#========================================================================================================
# NN Setup
#========================================================================================================
maxChannels = N_x // 2
model = Autoencoder1D(seq_len=dim, latent_dim=latentDim, max_channels=maxChannels, base_channels=baseChannels).to(DEVICE)
encoder_params = countParameters(model.encoder_conv) + countParameters(model.encoder_fc)
decoder_params = countParameters(model.decoder_fc) + countParameters(model.decoder_conv)
print(f"Encoder parameters: {encoder_params}")
print(f"Decoder parameters: {decoder_params}")

CR = U.size / (decoder_params + latentDim * U.shape[0] + 2)
print("Estimated CR=", CR)
if printModelFlag:
    # model = model.to('cpu')
    summary(model, (1, dim))
    # model = model.to(DEVICE)  # move it back to MPS or CUDA


outputFileName = "ModelTracer" + variable + "Nx" + str(N_x).zfill(4) + "Latent" + str(latentDim).zfill(2)
tracer = ModelTracer(model, torch.zeros(1, 1, dim).to(DEVICE))
print(tracer.summary())

df = tracer.to_dataframe()
print(df.head())
tracer.to_csv(saveModelTracerDirectory + outputFileName + ".csv")





criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-3)  # R: 1e-3
scheduler = LinearLR(optimizer, start_factor=1, end_factor=0.02, total_iters=450) # R: 1, 0.01. 450
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.5, patience=5, threshold=1e-4)

#========================================================================================================
# Training
#========================================================================================================
timeInitial = time.time()
trainLoss, validLoss, lr = [], [], []
epoch = 0
use_amp = (DEVICE.type == 'cuda')
scaler  = amp.GradScaler(enabled=use_amp)
trainer = CAE()
for epoch in range(1, N_epochs + 1):
    trainLoss.append(trainer.train(epoch))
    validLoss.append(trainer.validate(epoch))
    lr.append(scheduler.get_last_lr())
    # scheduler.step(validLoss[-1])  # Pass latest validation loss
    scheduler.step()
    # lr.append(optimizer.param_groups[0]['lr'])  # Log current learning rate

timeTraining = time.time() - timeInitial
print('Training time: %.2f' % timeTraining)

#========================================================================================================
# Performance Plots
#========================================================================================================
plotPerformance(trainLoss, validLoss, lr)

#========================================================================================================
# Save Model
#========================================================================================================
outputFileName = "model" + variable + "Nx" + str(N_x).zfill(4) + "Latent" + str(latentDim).zfill(2)
torch.save(model.state_dict(), saveModelDirectory + outputFileName + ".pth")
np.savez(saveParamsDirectory + outputFileName + ".npz",
         timeTraining=timeTraining, trainLoss=trainLoss, validLoss=validLoss, lr=lr,
         meanU=dataHandler.meanU_, stdU=dataHandler.stdU_)
