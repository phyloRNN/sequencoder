import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.backends.cuda

from utils import parse_alignment_file_gaps3D as parse_file

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from pathlib import Path
import time, sys

EPOCHS = 300
N_ALI_FILES = 10000
MAX_SITES = 1000
MAX_TAXA = 100

try:
    W_DIR = "/cluster/home/dsilvestro/sequencoder/"
    DATA_DIR = os.path.join(W_DIR, "omm_filtered_NT_CDS")
    RESULT_DIR = os.path.join(W_DIR, "results")
    os.makedirs(RESULT_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(RESULT_DIR, "axial_msa_transformer.pth")
except:
    W_DIR = None

DROP_GAPS = False
CHANNEL_DICT = ["A", "C", "G", "T"]

FINETUNE = False

LATENT_DIM = 128
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
DEBUG = False
TRAIN = True
predict_training_set = True
predict_test_set = True
device = "cuda" if torch.cuda.is_available() else "cpu"


class AxialAttentionBlock(nn.Module):
    """
    Alternates attention across rows (taxa) and columns (sites).
    Follows the norm-first approach for training stability.
    """

    def __init__(self, embed_dim, nhead, dropout=0.1):
        super().__init__()
        # Row (Taxa) Attention components
        self.row_norm = nn.LayerNorm(embed_dim)
        self.row_attn = nn.MultiheadAttention(
            embed_dim, nhead, dropout=dropout, batch_first=True
        )

        # Column (Site) Attention components
        self.col_norm = nn.LayerNorm(embed_dim)
        self.col_attn = nn.MultiheadAttention(
            embed_dim, nhead, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, row_mask=None, col_mask=None):
        """
        x: Tensor of shape [B, T, S, E] (Batch, Taxa, Sites, Embedding)
        row_mask: Padding mask for taxa rows [B, T]
        col_mask: Padding mask for site columns [B, S]
        """
        B, T, S, E = x.shape

        # --- 1. ROW ATTENTION (Across Taxa for each Site) ---
        res = x
        x_row = x.permute(0, 2, 1, 3).reshape(B * S, T, E)
        x_row = self.row_norm(x_row)

        if row_mask is not None:
            r_mask = row_mask.unsqueeze(1).expand(B, S, T).reshape(B * S, T)
        else:
            r_mask = None

        row_out, _ = self.row_attn(
            x_row, x_row, x_row, key_padding_mask=r_mask, need_weights=False
        )
        x = res + row_out.view(B, S, T, E).permute(0, 2, 1, 3)

        # --- 2. COLUMN ATTENTION (Across Sites for each Taxon) ---
        res = x
        x_col = x.reshape(B * T, S, E)
        x_col = self.col_norm(x_col)

        if col_mask is not None:
            c_mask = col_mask.unsqueeze(1).expand(B, T, S).reshape(B * T, S)
        else:
            c_mask = None

        col_out, _ = self.col_attn(
            x_col, x_col, x_col, key_padding_mask=c_mask, need_weights=False
        )
        x = res + col_out.view(B, T, S, E)

        # --- 3. FEED-FORWARD NETWORK ---
        res = x
        x_ffn = self.ffn_norm(x)
        x = res + self.ffn(x_ffn)

        return x


class AxialMSATransformer(nn.Module):
    def __init__(
        self,
        in_channels=4,
        embed_dim=64,
        latent_dim=128,
        nhead=4,
        num_layers=3,
        max_sites=10000,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        # Encoder Projections
        self.token_embeddings = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.pos_embedding = nn.Embedding(max_sites, embed_dim)

        self.encoder_layers = nn.ModuleList(
            [AxialAttentionBlock(embed_dim, nhead) for _ in range(num_layers)]
        )
        self.to_latent = nn.Linear(embed_dim * 2, latent_dim)

        # NEW AXIAL DECODER COMPONENTS
        self.from_latent = nn.Linear(latent_dim, embed_dim)
        self.decoder_token_embeddings = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        self.decoder_layers = nn.ModuleList(
            [AxialAttentionBlock(embed_dim, nhead) for _ in range(num_layers)]
        )
        self.decoder_head = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

    def encode(self, x, row_mask=None, col_mask=None):
        B, C, T, S = x.shape
        device = x.device

        if row_mask is None or col_mask is None:
            with torch.no_grad():
                col_mask = x.sum(dim=(1, 2)) == 0  # [B, S]
                row_mask = x.sum(dim=(1, 3)) == 0  # [B, T]

        x_emb = self.token_embeddings(x)  # [B, E, T, S]
        x_emb = x_emb.permute(0, 2, 3, 1)  # [B, T, S, E]

        positions = (
            torch.arange(0, S, device=device).unsqueeze(0).unsqueeze(1)
        )  # [1, 1, S]
        x_emb = x_emb + self.pos_embedding(positions)

        for layer in self.encoder_layers:
            x_emb = layer(x_emb, row_mask=row_mask, col_mask=col_mask)

        if row_mask is not None:
            x_emb = x_emb.masked_fill(row_mask.unsqueeze(2).unsqueeze(3), 0.0)
        if col_mask is not None:
            x_emb = x_emb.masked_fill(col_mask.unsqueeze(1).unsqueeze(3), 0.0)

        # Symmetric Taxa Pooling
        valid_rows = (~row_mask).sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1.0)
        row_mean = x_emb.sum(dim=1) / valid_rows  # [B, S, E]

        x_for_max = (
            x_emb.masked_fill(row_mask.unsqueeze(2).unsqueeze(3), -1e9)
            if row_mask is not None
            else x_emb
        )
        row_max = x_for_max.max(dim=1)[0]  # [B, S, E]

        site_features = torch.cat([row_mean, row_max], dim=-1)  # [B, S, E * 2]

        # Symmetric Site Pooling
        valid_cols = (~col_mask).sum(dim=1, keepdim=True).clamp(min=1.0)
        site_mean = site_features.sum(dim=1) / valid_cols  # [B, E * 2]

        latent = self.to_latent(site_mean)  # [B, Latent_Dim]
        return latent

    def forward(self, x, mask_ratio=0.15):
        B, C, T, S = x.shape
        device = x.device

        # 1. Compute true padding masks from clean structural data before masking
        with torch.no_grad():
            col_mask = x.sum(dim=(1, 2)) == 0  # [B, S]
            row_mask = x.sum(dim=(1, 3)) == 0  # [B, T]

        # 2. Perform Masked Site Modeling Column Operations
        masked_x = x.clone()
        if mask_ratio > 0:
            rand_mask = torch.rand(B, S, device=device) < mask_ratio
            non_padded_columns = ~col_mask  # [B, S]
            final_mask = rand_mask & non_padded_columns  # [B, S]

            mask_expanded = final_mask.unsqueeze(1).unsqueeze(2).expand(-1, C, T, -1)
            masked_x[mask_expanded] = 0.0
        else:
            final_mask = torch.zeros((B, S), dtype=torch.bool, device=device)
            mask_expanded = final_mask.unsqueeze(1).unsqueeze(2).expand(-1, C, T, -1)

        # 3. Process features through invariant Encoder pipeline
        latent = self.encode(
            masked_x, row_mask=row_mask, col_mask=col_mask
        )  # [B, Latent_Dim]

        # 4. Process features through equivariant Axial Decoder pipeline
        recon_sig = self.from_latent(latent)  # [B, E]
        latent_expanded = (
            recon_sig.unsqueeze(1).unsqueeze(2).expand(-1, T, S, -1)
        )  # [B, T, S, E]

        # Map masked input to provide Identity Anchors per row
        decoder_features = self.decoder_token_embeddings(masked_x)  # [B, E, T, S]
        decoder_features = decoder_features.permute(0, 2, 3, 1)  # [B, T, S, E]

        # Horizontal Positional Embeddings
        positions = (
            torch.arange(0, S, device=device).unsqueeze(0).unsqueeze(1)
        )  # [1, 1, S]
        pos_emb = self.pos_embedding(positions)

        # Combine streams
        dec_in = decoder_features + latent_expanded + pos_emb

        # Axial Self-Attention Decoder loop
        for layer in self.decoder_layers:
            dec_in = layer(dec_in, row_mask=row_mask, col_mask=col_mask)

        # Final Convolution Mapping to channel dimensions
        fused = dec_in.permute(0, 3, 1, 2)  # [B, E, T, S]
        recon_logits = self.decoder_head(fused)  # [B, C, T, S]

        return recon_logits, latent, mask_expanded


class SeqBinaryFileDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data_np = parse_file(file_path, drop_gaps=DROP_GAPS, channel_dict=CHANNEL_DICT)

        if data_np.shape[2] > MAX_SITES:
            start = np.random.randint(0, data_np.shape[-1] - MAX_SITES)
            data_np = data_np[:, :, start : start + MAX_SITES]

        if data_np.shape[1] > MAX_TAXA:
            indices = np.random.choice(data_np.shape[1], MAX_TAXA, replace=False)
            data_np = data_np[:, indices, :]

        data_tensor = torch.from_numpy(data_np).float()
        return data_tensor


def variable_msa_collate_fn(batch):
    max_channels = batch[0].shape[0]
    max_taxa = max([sample.shape[1] for sample in batch])
    max_sites = max([sample.shape[2] for sample in batch])

    padded_batch = []
    for sample in batch:
        C, T, S = sample.shape
        padded_sample = torch.zeros((C, max_taxa, max_sites), dtype=sample.dtype)
        padded_sample[:, :T, :S] = sample
        padded_batch.append(padded_sample)

    return torch.stack(padded_batch, dim=0)


def get_channel_densities(file_path, schema="fasta"):
    data = parse_file(
        file_path, schema=schema, drop_gaps=DROP_GAPS, channel_dict=CHANNEL_DICT
    )
    densities = data.mean(axis=(1, 2))
    densities = np.append(densities, data.shape[-1])
    densities = np.append(densities, int(data.shape[-1] > MAX_SITES))
    return densities


if __name__ == "__main__":
    if TRAIN:
        files = glob.glob(os.path.join(DATA_DIR, "*"))[:N_ALI_FILES]
        dataset = SeqBinaryFileDataset(files)

        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_subset, val_subset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=variable_msa_collate_fn,
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=variable_msa_collate_fn,
        )

        model = AxialMSATransformer(
            in_channels=len(CHANNEL_DICT), latent_dim=LATENT_DIM, max_sites=MAX_SITES
        )

        if FINETUNE:
            model.load_state_dict(torch.load("best_model.pth", map_location=device))
            print("\nModel loaded successfully!", flush=True)

        model.to(device)

        INITIAL_LR = 1e-4 if FINETUNE else 2e-4
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=INITIAL_LR, weight_decay=1e-4
        )

        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

        patience = 15
        counter = 0
        best_val_loss = 0.5444 if FINETUNE else float("inf")

        # Track history variables
        history_data = []
        history_csv_path = os.path.join(RESULT_DIR, "training_history.csv")

        for epoch in range(EPOCHS):
            model.train()
            running_train_recon_loss = 0.0
            running_train_decorr_loss = 0.0
            lambda_decorr = 0.01

            diag_mask = torch.eye(LATENT_DIM).to(device)
            latent_buffer = []

            optimizer.zero_grad()

            for batch_n, batch in enumerate(train_loader, 1):
                batch = batch.to(device)

                reconstruction, latent, mask_indices = model(batch, mask_ratio=0.15)

                raw_recon_loss = criterion(reconstruction, batch)
                valid_data_mask = batch.sum(dim=1, keepdim=True) > 0
                final_loss_mask = mask_indices & valid_data_mask

                if final_loss_mask.sum() > 0:
                    recon_loss = raw_recon_loss[final_loss_mask].mean()
                else:
                    recon_loss = raw_recon_loss.mean()

                d_loss = torch.tensor(0.0).to(device)
                if len(latent_buffer) > 0:
                    past_latents = torch.cat(latent_buffer, dim=0)
                    z_combined = torch.cat([latent, past_latents], dim=0)

                    mu = z_combined.mean(dim=0)
                    std = z_combined.std(dim=0) + 1e-8
                    z_std = (z_combined - mu) / std

                    corr_matrix = (z_std.T @ z_std) / (z_std.size(0) - 1)
                    off_diagonals = corr_matrix * (1 - diag_mask)
                    d_loss = off_diagonals.abs().mean()

                total_batch_loss = recon_loss + (d_loss * lambda_decorr)
                total_batch_loss = total_batch_loss / GRADIENT_ACCUMULATION_STEPS
                total_batch_loss.backward()

                if batch_n % GRADIENT_ACCUMULATION_STEPS == 0 or batch_n == len(
                    train_loader
                ):
                    optimizer.step()
                    optimizer.zero_grad()

                with torch.no_grad():
                    latent_buffer.append(latent.detach())
                    if len(latent_buffer) > 16:
                        latent_buffer.pop(0)

                if batch_n % 100 == 0 or batch_n == len(train_loader):
                    with torch.no_grad():
                        all_latents = torch.cat(latent_buffer, dim=0)
                        std_per_dim = all_latents.std(dim=0)
                        active_dims = (std_per_dim > 1e-4).sum().item()
                        mean_corr = (
                            off_diagonals.abs().mean().item()
                            if len(latent_buffer) > 1
                            else 0
                        )

                    s = (
                        f"Epoch {epoch + 1}, Batch {batch_n}/{len(train_loader)} | "
                        f"MSM-Loss: {recon_loss.item():.4f} | "
                        f"D-Loss: {d_loss.item():.4f} | "
                        f"Avg Corr: {mean_corr:.2f} | "
                        f"Active Dims: {active_dims}/{LATENT_DIM}"
                    )
                    print(s, flush=True)

                running_train_recon_loss += recon_loss.item()
                running_train_decorr_loss += d_loss.item()

            # --- VALIDATION PHASE ---
            model.eval()
            running_val_recon_loss = 0.0
            running_val_decorr_loss = 0.0
            val_latent_buffer = []

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)

                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        reconstruction, latent, mask_indices = model(
                            batch, mask_ratio=0.15
                        )
                        raw_v_loss = criterion(reconstruction, batch)
                        valid_data_mask = batch.sum(dim=1, keepdim=True) > 0
                        final_loss_mask = mask_indices & valid_data_mask

                        if final_loss_mask.sum() > 0:
                            v_loss = raw_v_loss[final_loss_mask].mean()
                        else:
                            v_loss = raw_v_loss.mean()

                        # Evaluate validation batch decorrelation
                        v_d_loss = torch.tensor(0.0).to(device)
                        if len(val_latent_buffer) > 0:
                            past_val_latents = torch.cat(val_latent_buffer, dim=0)
                            z_combined_val = torch.cat(
                                [latent, past_val_latents], dim=0
                            )

                            mu_v = z_combined_val.mean(dim=0)
                            std_v = z_combined_val.std(dim=0) + 1e-8
                            z_std_v = (z_combined_val - mu_v) / std_v

                            corr_matrix_v = (z_std_v.T @ z_std_v) / (
                                z_std_v.size(0) - 1
                            )
                            off_diagonals_v = corr_matrix_v * (1 - diag_mask)
                            v_d_loss = off_diagonals_v.abs().mean()

                    running_val_recon_loss += v_loss.item()
                    running_val_decorr_loss += v_d_loss.item()

                    val_latent_buffer.append(latent.detach())
                    if len(val_latent_buffer) > 16:
                        val_latent_buffer.pop(0)

            avg_train_recon = running_train_recon_loss / len(train_loader)
            avg_train_decorr = running_train_decorr_loss / len(train_loader)
            avg_val_recon = running_val_recon_loss / len(val_loader)
            avg_val_decorr = running_val_decorr_loss / len(val_loader)
            avg_val_total = avg_val_recon + (avg_val_decorr * lambda_decorr)

            print(
                f"\n--- Epoch {epoch + 1} Complete ---"
                f"\nTrain MSM-Loss: {avg_train_recon:.4f} | Train D-Loss: {avg_train_decorr:.4f}"
                f"\nVal MSM-Loss:   {avg_val_recon:.4f} | Val D-Loss:   {avg_val_decorr:.4f}",
                flush=True,
            )

            # Record history snapshot
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_msm_loss": avg_train_recon,
                "train_d_loss": avg_train_decorr,
                "val_msm_loss": avg_val_recon,
                "val_d_loss": avg_val_decorr,
            }
            history_data.append(epoch_metrics)

            # Write out file directly to safe path
            df_history = pd.DataFrame(history_data)
            df_history.to_csv(history_csv_path, index=False)

            # Early Stopping metric checkpointing guided by reconstruction target
            if avg_val_total < best_val_loss:
                best_val_loss = avg_val_total
                counter = 0
                torch.save(model.state_dict(), "best_model.pth")
                print("★ Performance improved! Saving checkpoint.", flush=True)
            else:
                counter += 1
                print(
                    f"No improvement. EarlyStopping: {counter}/{patience}", flush=True
                )
                if counter >= patience:
                    print("Early stopping triggered. Terminating loop.", flush=True)
                    break

        # Plot loss paths when metrics loop terminates
        if len(history_data) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.plot(
                df_history["epoch"],
                df_history["train_msm_loss"],
                label="Train MSM Loss",
                color="royalblue",
            )
            ax1.plot(
                df_history["epoch"],
                df_history["val_msm_loss"],
                label="Val MSM Loss",
                color="darkorange",
            )
            ax1.set_title("Masked Site Modeling (MSM) Reconstruction Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True, linestyle="--", alpha=0.5)

            ax2.plot(
                df_history["epoch"],
                df_history["train_d_loss"],
                label="Train D-Loss",
                color="forestgreen",
            )
            ax2.plot(
                df_history["epoch"],
                df_history["val_d_loss"],
                label="Val D-Loss",
                color="crimson",
            )
            ax2.set_title("Latent Space Decorrelation (D) Loss")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss")
            ax2.legend()
            ax2.grid(True, linestyle="--", alpha=0.5)

            fig.tight_layout()
            fig.savefig(os.path.join(RESULT_DIR, "loss_curves.pdf"), dpi=300)
            plt.close(fig)
            print(
                f"Loss curves saved to {os.path.join(RESULT_DIR, 'loss_curves.pdf')}",
                flush=True,
            )

        model.load_state_dict(torch.load("best_model.pth"))
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Final training weights committed to {MODEL_PATH}", flush=True)

    else:
        model = AxialMSATransformer(
            latent_dim=LATENT_DIM, in_channels=len(CHANNEL_DICT), max_sites=MAX_SITES
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print("\nModel loaded successfully!", flush=True)
        files = glob.glob(os.path.join(DATA_DIR, "*"))[:N_ALI_FILES]

    if predict_training_set:
        all_embeddings = []
        files = glob.glob(os.path.join(DATA_DIR, "*"))[:N_ALI_FILES]

        i = 0
        with torch.no_grad():
            for f_path in files:
                data_np = parse_file(
                    f_path, drop_gaps=DROP_GAPS, channel_dict=CHANNEL_DICT
                )

                if data_np.shape[2] > MAX_SITES:
                    data_np = data_np[:, :, :MAX_SITES]

                data = torch.from_numpy(data_np).float().unsqueeze(0).to(device)
                latent = model.encode(data)
                all_embeddings.append(latent.squeeze().cpu().numpy())
                i += 1

        matrix = np.array(all_embeddings)

        all_densities = []
        for f_path in files:
            all_densities.append(get_channel_densities(f_path))
        density_matrix = np.array(all_densities)

        data_to_save = {"file_name": [os.path.basename(f) for f in files]}
        for i in range(matrix.shape[1]):
            data_to_save[f"dim_{i}"] = matrix[:, i]
        for i in range(density_matrix.shape[1]):
            data_to_save[f"density_ch_{i}"] = density_matrix[:, i]

        df = pd.DataFrame(data_to_save)
        df.to_csv(os.path.join(RESULT_DIR, "embeddings_results.csv"), index=False)
        print("\nSaved embeddings to embeddings_results.csv", flush=True)

    if predict_test_set:
        test_files = glob.glob(os.path.join(DATA_DIR, "*"))[N_ALI_FILES:]

        all_densities = []
        for f_path in test_files:
            all_densities.append(get_channel_densities(f_path))
        density_matrix = np.array(all_densities)

        all_embeddings = []
        model.eval()
        i = 0
        with torch.no_grad():
            for f_path in test_files:
                data_np = parse_file(
                    f_path, drop_gaps=DROP_GAPS, channel_dict=CHANNEL_DICT
                )

                if data_np.shape[2] > MAX_SITES:
                    data_np = data_np[:, :, :MAX_SITES]

                data = torch.from_numpy(data_np).float().unsqueeze(0).to(device)
                latent = model.encode(data)
                all_embeddings.append(latent.squeeze().cpu().numpy())
                i += 1

        matrix = np.array(all_embeddings)

        data_to_save = {"file_name": [os.path.basename(f) for f in test_files]}
        for i in range(matrix.shape[1]):
            data_to_save[f"dim_{i}"] = matrix[:, i]
        for i in range(density_matrix.shape[1]):
            data_to_save[f"density_ch_{i}"] = density_matrix[:, i]

        df = pd.DataFrame(data_to_save)
        df.to_csv(os.path.join(RESULT_DIR, "embeddings_results_test.csv"), index=False)
        print("\nSaved embeddings to embeddings_results_test.csv", flush=True)

    # --- UMAP & PLOTTING PIPELINE ---
    f = os.path.join(RESULT_DIR, "embeddings_results.csv")
    data = pd.read_csv(f)
    latent_cols = [f"dim_{i}" for i in range(128)]
    X_train = data[latent_cols].values

    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=123
    )
    embedding_train = reducer.fit_transform(X_train)

    data["umap_0"] = embedding_train[:, 0]
    data["umap_1"] = embedding_train[:, 1]

    f = os.path.join(RESULT_DIR, "embeddings_results_test.csv")
    data_test = pd.read_csv(f)
    X_test = data_test[latent_cols].values

    embedding_test = reducer.transform(X_test)
    data_test["umap_0"] = embedding_test[:, 0]
    data_test["umap_1"] = embedding_test[:, 1]

    channel_names = [
        "freq. A",
        "freq C ",
        "freq. T",
        "freq. G",
        "ali. length",
        "oversize ali",
    ]
    if "-" in CHANNEL_DICT and not DROP_GAPS:
        channel_names.append("freq. gaps")

    n_cols = len(channel_names)
    fig_umap, axes_umap = plt.subplots(2, n_cols, figsize=(n_cols * 5.3, 10))
    fig_hist, axes_hist = plt.subplots(2, n_cols, figsize=(n_cols * 4.2, 6))

    for i in range(n_cols):
        if channel_names[i] == "ali. length":
            c_train = np.log10(data[f"density_ch_{i}"])
            label = "log10(ali. length)"
        else:
            c_train = data[f"density_ch_{i}"]
            label = channel_names[i]

        scatter0 = axes_umap[0, i].scatter(
            data["umap_0"], data["umap_1"], c=c_train, cmap="viridis", s=10, alpha=0.6
        )
        axes_umap[0, i].set_title(f"Train: {channel_names[i]}")
        scatter0.set_alpha(1.0)
        fig_umap.colorbar(scatter0, ax=axes_umap[0, i])
        scatter0.set_alpha(0.7)

        axes_hist[0, i].hist(
            c_train, bins=30, color="skyblue", edgecolor="black", alpha=0.7
        )
        axes_hist[0, i].set_title(f"Train Dist: {label}")
        axes_hist[0, i].set_ylabel("Count")

        if channel_names[i] == "ali. length":
            c_test = np.log10(data_test[f"density_ch_{i}"])
        else:
            c_test = data_test[f"density_ch_{i}"]

        scatter1 = axes_umap[1, i].scatter(
            data_test["umap_0"],
            data_test["umap_1"],
            c=c_test,
            cmap="viridis",
            s=10,
            alpha=0.6,
        )
        axes_umap[1, i].set_title(f"Test: {channel_names[i]}")
        scatter1.set_alpha(1.0)
        fig_umap.colorbar(scatter1, ax=axes_umap[1, i])
        scatter1.set_alpha(0.7)

        axes_hist[1, i].hist(
            c_test, bins=30, color="salmon", edgecolor="black", alpha=0.7
        )
        axes_hist[1, i].set_title(f"Test Dist: {label}")
        axes_hist[1, i].set_ylabel("Count")

    fig_umap.tight_layout()
    fig_umap.savefig(
        os.path.join(RESULT_DIR, "umap_test_projection.png"),
        dpi=300,
        bbox_inches="tight",
    )

    fig_hist.tight_layout()
    fig_hist.savefig(
        os.path.join(RESULT_DIR, "density_distributions.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig_umap)
    plt.close(fig_hist)
