import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.backends.cuda

from utils import parse_alignment_file_gaps3D as parse_file

# from utils import print_update
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from pathlib import Path
import time, tifffile, sys

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
CHANNEL_DICT = ["A", "C", "G", "T"]  # , "-"

FINETUNE = True

LATENT_DIM = 128
BATCH_SIZE = 8  # Reduced from 16
GRADIENT_ACCUMULATION_STEPS = 4  # 4 * 4 = effective batch size of 16
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

        self.token_embeddings = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.pos_embedding = nn.Embedding(max_sites, embed_dim)

        self.encoder_layers = nn.ModuleList(
            [AxialAttentionBlock(embed_dim, nhead) for _ in range(num_layers)]
        )

        self.to_latent = nn.Linear(embed_dim * 2, latent_dim)
        self.from_latent = nn.Linear(latent_dim, embed_dim)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder_transformer = nn.TransformerEncoder(
            decoder_layer, num_layers=num_layers
        )

        self.lineage_projector = nn.Sequential(
            nn.Linear(in_channels, 16), nn.GELU(), nn.Linear(16, 16)
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim + 16, embed_dim), nn.GELU()
        )
        self.decoder_head = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

    def encode(self, x):
        B, C, T, S = x.shape
        device = x.device

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

        row_mean = x_emb.sum(dim=1) / (~row_mask).sum(dim=1, keepdim=True).unsqueeze(
            -1
        ).clamp(
            min=1.0
        )  # [B, S, E]
        row_max = x_emb.max(dim=1)[0]  # [B, S, E]
        site_features = torch.cat([row_mean, row_max], dim=-1)  # [B, S, E * 2]

        site_mean = site_features.sum(dim=1) / (~col_mask).sum(
            dim=1, keepdim=True
        ).clamp(
            min=1.0
        )  # [B, E * 2]

        latent = self.to_latent(site_mean)  # [B, Latent_Dim]
        return latent

    def forward(self, x):
        B, C, T, S = x.shape
        device = x.device

        latent = self.encode(x)  # [B, Latent_Dim]

        with torch.no_grad():
            valid_sites_mask = (x.sum(dim=1, keepdim=True) > 0).float()
            site_counts = valid_sites_mask.sum(dim=3, keepdim=True).clamp(min=1.0)
            col_mask = x.sum(dim=(1, 2)) == 0

        raw_lineage_profiles = (x * valid_sites_mask).sum(dim=3) / site_counts.squeeze(
            -1
        )
        lineage_features = self.lineage_projector(
            raw_lineage_profiles.transpose(1, 2)
        )  # [B, T, 16]

        recon_sig = self.from_latent(latent).unsqueeze(1).expand(-1, S, -1)  # [B, S, E]
        positions = torch.arange(0, S, device=device).unsqueeze(0)
        recon_sites = recon_sig + self.pos_embedding(positions)

        decoded_sites = self.decoder_transformer(
            recon_sites, src_key_padding_mask=col_mask
        )
        decoded_sites = decoded_sites.transpose(1, 2)  # [B, E, S]

        expanded_sites = decoded_sites.unsqueeze(2).expand(
            -1, -1, T, -1
        )  # [B, E, T, S]
        expanded_lineage = (
            lineage_features.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, S)
        )  # [B, 16, T, S]

        combined = torch.cat([expanded_sites, expanded_lineage], dim=1).permute(
            0, 2, 3, 1
        )  # [B, T, S, E + 16]
        fused = self.fusion_layer(combined).permute(0, 3, 1, 2)  # [B, E, T, S]

        recon_logits = self.decoder_head(fused)  # [B, C, T, S]

        # FIX: Removed torch.sigmoid() here. Returning raw logits for BCEWithLogitsLoss compatibility.
        return recon_logits, latent


def apply_column_mask(x, mask_ratio=0.15):
    B, C, T, S = x.shape
    masked_x = x.clone()

    rand_mask = torch.rand(B, S, device=x.device) < mask_ratio
    non_padded_columns = x.sum(dim=(1, 2)) > 0  # Shape: [B, S]
    final_mask = rand_mask & non_padded_columns  # Shape: [B, S]

    mask_expanded = final_mask.unsqueeze(1).unsqueeze(2).expand(-1, C, T, -1)
    masked_x[mask_expanded] = 0.0

    return masked_x, mask_expanded


class SeqBinaryFileDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data_np = parse_file(file_path, drop_gaps=DROP_GAPS, channel_dict=CHANNEL_DICT)

        # Trim sites if too long
        if data_np.shape[2] > MAX_SITES:
            start = np.random.randint(0, data_np.shape[-1] - MAX_SITES)
            data_np = data_np[:, :, start : start + MAX_SITES]

        # --- ADD THIS: Trim/Sub-sample taxa if there are too many sequences ---
        if data_np.shape[1] > MAX_TAXA:
            # Randomly select subset of sequences to maintain diversity
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

        # 2. Re-initialize the optimizer with a lower learning rate (Safety Cushion)
        INITIAL_LR = (
            1e-4 if FINETUNE else 1e-3
        )  # Dropped from 1e-3 to 1e-4 for resuming
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=INITIAL_LR, weight_decay=1e-4
        )

        # FIX: Swapped BCELoss to BCEWithLogitsLoss to bypass unsafe autocast constraints safely
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

        patience = 5
        counter = 0
        best_val_loss = 0.5444 if FINETUNE else float("inf")

        for epoch in range(EPOCHS):
            model.train()
            running_train_loss = 0.0
            lambda_decorr = 0.01

            diag_mask = torch.eye(LATENT_DIM).to(device)
            latent_buffer = []

            optimizer.zero_grad()  # Reset BEFORE the batch loop starts

            for batch_n, batch in enumerate(train_loader, 1):
                batch = batch.to(device)

                # Forward pass & loss computation
                masked_batch, mask_indices = apply_column_mask(batch, mask_ratio=0.15)
                reconstruction, latent = model(masked_batch)
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

                # --- MODIFIED: Normalize loss by accumulation steps ---
                total_batch_loss = total_batch_loss / GRADIENT_ACCUMULATION_STEPS
                total_batch_loss.backward()

                # --- MODIFIED: Only step when steps are reached ---
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

                running_train_loss += recon_loss.item()

            # --- VALIDATION PHASE ---
            model.eval()
            running_val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    masked_batch, mask_indices = apply_column_mask(
                        batch, mask_ratio=0.15
                    )

                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        reconstruction, _ = model(masked_batch)
                        raw_v_loss = criterion(reconstruction, batch)
                        valid_data_mask = batch.sum(dim=1, keepdim=True) > 0
                        final_loss_mask = mask_indices & valid_data_mask

                        if final_loss_mask.sum() > 0:
                            v_loss = raw_v_loss[final_loss_mask].mean()
                        else:
                            v_loss = raw_v_loss.mean()

                    running_val_loss += v_loss.item()

            avg_train_loss = running_train_loss / len(train_loader)
            avg_val_loss = running_val_loss / len(val_loader)

            print(
                f"\n--- Epoch {epoch + 1} Complete ---"
                f"\nTrain MSM-Loss: {avg_train_loss:.4f} | Val MSM-Loss: {avg_val_loss:.4f}",
                flush=True,
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
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

                # print_update(f"Processed: {os.path.basename(f_path)}, {i}")
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
                # print_update(f"Processed: {os.path.basename(f_path)}, {i}")
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
