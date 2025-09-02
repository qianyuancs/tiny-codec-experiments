# tiny_codec.py
# Minimal neural "codec" demo (Encoder -> VQ -> Decoder)
# CPU friendly, meant for small datasets (e.g., 20 wavs).
# Usage:
#   python tiny_codec.py --data_dir /path/to/wavs --max_steps 600 --sample_rate 16000

import os
import glob
import random
import argparse
from pathlib import Path
import math
import csv
import time
import pandas as pd

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

# Optional metrics (ensure installed)
try:
    from pystoi import stoi
except Exception:
    stoi = None
try:
    from pesq import pesq
except Exception:
    pesq = None

# -----------------------
# Dataset
# -----------------------
class WavDataset(Dataset):
    def __init__(self, folder, sample_rate=16000, segment_seconds=1.0):
        files = sorted(glob.glob(os.path.join(folder, "*.wav")))
        if len(files) == 0:
            raise RuntimeError(f"No wav files found in {folder}")
        self.files = files
        self.sr = sample_rate
        self.segment_len = int(segment_seconds * sample_rate)
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        p = self.files[idx]
        wav, sr = torchaudio.load(p)  # shape (C, L)
        wav = wav.mean(0)  # mono
        wav = wav.numpy().astype(np.float32)
        if sr != self.sr:
            wav = torchaudio.transforms.Resample(sr, self.sr)(torch.from_numpy(wav)).numpy()
        L = wav.shape[0]
        if L < self.segment_len:
            # pad
            pad = self.segment_len - L
            wav = np.pad(wav, (0, pad), mode='constant', constant_values=0.0)
        else:
            # random crop
            start = random.randint(0, L - self.segment_len)
            wav = wav[start:start + self.segment_len]
        wav = torch.from_numpy(wav).unsqueeze(0)  # (1, segment_len)
        return wav, os.path.basename(p)

# -----------------------
# Tiny Encoder / Decoder
# -----------------------
class TinyEncoder(nn.Module):
    def __init__(self, in_ch=1, hidden=32, z_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, hidden, kernel_size=15, stride=2, padding=7)
        self.conv2 = nn.Conv1d(hidden, hidden*2, kernel_size=15, stride=2, padding=7)
        self.conv3 = nn.Conv1d(hidden*2, z_dim, kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU(0.2)
    def forward(self, x):
        # x: (B,1,L)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        z = self.conv3(x)
        return z  # (B, z_dim, T)  T = L/4

class TinyDecoder(nn.Module):
    def __init__(self, out_ch=1, hidden=32, z_dim=64):
        super().__init__()
        self.deconv1 = nn.ConvTranspose1d(z_dim, hidden*2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose1d(hidden*2, hidden, kernel_size=4, stride=2, padding=1)
        self.out = nn.Conv1d(hidden, out_ch, kernel_size=7, padding=3)
        self.act = nn.LeakyReLU(0.2)
    def forward(self, z):
        x = self.act(self.deconv1(z))
        x = self.act(self.deconv2(x))
        x = torch.tanh(self.out(x))
        return x  # (B,1,L)

# -----------------------
# Simple VQ (non-EMA)
# -----------------------
class VQEmbedding(nn.Module):
    def __init__(self, num_embeddings=256, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        self.commitment_cost = commitment_cost
    def forward(self, inputs):
        # inputs: (B, D, T)
        B, D, T = inputs.shape
        # flatten -> (B*T, D)
        flat = inputs.permute(0,2,1).contiguous().view(-1, D)
        # distances (||x - e||^2) efficient computation
        # dist = x^2 + e^2 - 2 x e^T
        e = self.embedding.weight  # (K,D)
        # compute squared distances
        # flat_sq (N,1), e_sq (K,), cross (N,K)
        flat_sq = torch.sum(flat**2, dim=1, keepdim=True)
        e_sq = torch.sum(e**2, dim=1)
        cross = torch.matmul(flat, e.t())
        dists = flat_sq + e_sq.unsqueeze(0) - 2.0 * cross  # (N, K)
        encoding_indices = torch.argmin(dists, dim=1)  # (N,)
        quantized = self.embedding(encoding_indices).view(B, T, D).permute(0,2,1).contiguous()
        # losses
        # codebook loss
        codebook_loss = F.mse_loss(quantized.detach(), inputs)
        commitment_loss = F.mse_loss(quantized, inputs.detach())
        loss_vq = codebook_loss + self.commitment_cost * commitment_loss
        # straight-through estimator
        quantized_st = inputs + (quantized - inputs).detach()
        # reshape indices (B, T)
        indices = encoding_indices.view(B, T)
        return quantized_st, loss_vq, indices

# -----------------------
# Full model wrapper
# -----------------------
class TinyCodec(nn.Module):
    def __init__(self, z_dim=64, codebook_size=256, commitment_cost=0.25):
        super().__init__()
        self.encoder = TinyEncoder(in_ch=1, hidden=32, z_dim=z_dim)
        self.vq = VQEmbedding(num_embeddings=codebook_size, embedding_dim=z_dim, commitment_cost=commitment_cost)
        self.decoder = TinyDecoder(out_ch=1, hidden=32, z_dim=z_dim)
    def forward(self, x):
        z = self.encoder(x)          # (B, D, T)
        quantized, vq_loss, indices = self.vq(z)
        recon = self.decoder(quantized)
        return recon, vq_loss, indices

# -----------------------
# Utility metrics
# -----------------------
def snr_db(ref, deg, eps=1e-9):
    # ref/deg: numpy 1d
    noise = ref - deg
    p_sig = np.sum(ref.astype(np.float64)**2)
    p_noise = np.sum(noise.astype(np.float64)**2) + eps
    return 10.0 * np.log10((p_sig + eps) / p_noise)

# -----------------------
# Train loop
# -----------------------
def train(args):
    device = torch.device("cpu")
    torch.set_num_threads(2)  # limit threads for CPU
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds = WavDataset(args.data_dir, sample_rate=args.sample_rate, segment_seconds=args.segment_seconds)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = TinyCodec(z_dim=args.z_dim, codebook_size=args.codebook_size, commitment_cost=args.commitment_cost).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = results_dir / "recon"
    recon_dir.mkdir(exist_ok=True)

    metrics_file = results_dir / "metrics.csv"
    csvfile = open(metrics_file, "w", newline='', encoding='utf-8')
    writer = csv.writer(csvfile)
    writer.writerow(["step", "file", "snr_db", "stoi", "pesq"])

    step = 0
    start = time.time()
    loss_hist = []
    while step < args.max_steps:
        for batch in loader:
            model.train()
            x, name = batch
            x = x.to(device)  # (B,1,L)
            optimizer.zero_grad()
            recon, vq_loss, indices = model(x)
            recon_loss = F.l1_loss(recon, x)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()

            loss_hist.append(loss.item())
            if step % args.log_interval == 0:
                elapsed = time.time() - start
                avg_loss = sum(loss_hist[-50:]) / max(1, min(len(loss_hist), 50))
                print(f"[{step}/{args.max_steps}] loss={loss.item():.4f} recon={recon_loss.item():.4f} vq={vq_loss.item():.4f} avg50={avg_loss:.4f} elapsed={elapsed:.1f}s")

            # periodically save reconstructions & evaluate small batch
            if step % args.save_interval == 0:
                model.eval()
                with torch.no_grad():
                    # take first sample in batch
                    ref = x[0].cpu().numpy().squeeze()  # (L,)
                    out = recon[0].cpu().numpy().squeeze()
                    # save wav (clamp)
                    out_clamped = np.clip(out, -1.0, 1.0)
                    fn = recon_dir / f"recon_step{step:06d}_{name[0]}"
                    sf.write(str(fn), out_clamped, args.sample_rate)
                    # compute metrics if packages available
                    try:
                        s = snr_db(ref, out_clamped)
                    except Exception:
                        s = None
                    try:
                        stoi_v = stoi(ref, out_clamped, args.sample_rate) if stoi is not None else None
                    except Exception:
                        stoi_v = None
                    try:
                        pesq_v = pesq(args.sample_rate, ref, out_clamped, 'wb') if pesq is not None else None
                    except Exception:
                        pesq_v = None
                    writer.writerow([step, name[0], s, stoi_v, pesq_v])
                    csvfile.flush()
                    print(f"  Saved recon to {fn}, SNR={s}, STOI={stoi_v}, PESQ={pesq_v}")

            step += 1
            if step >= args.max_steps:
                break
    csvfile.close()
    # final save model
    torch.save(model.state_dict(), "results/tiny_codec_final.pt")
    loss_df = pd.DataFrame({"step": list(range(len(loss_hist))), "loss": loss_hist})
    loss_df.to_csv("results/loss.600.csv", index=False)
    print("Saved loss history to results/loss.csv")
    print("Training finished. Model saved to results/tiny_codec_final.pt")

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="folder with wav files")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--segment_seconds", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    print("ARGS:", args)
    train(args)
