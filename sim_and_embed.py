import os

# from te .nsorboard.compat.tensorflow_stub.io.gfile import exists

# from sequencoder_no_gap import *
from sequencoder_transformer_coaxial import *
from utils.utilities import *
from utils.simulate import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
import seaborn as sns

####------- CHECK AGAINST SIMULATIONS ---------####
W_DIR = "/Users/dsilvestro/Documents/Projects/Ongoing/GenAli"
bin_dir = "/Users/dsilvestro/Software/phyloRNN-project/phyloRNN/phyloRNN/bin/"
msa_dir = os.path.join(W_DIR, "OrthoMamv12_nogaps/omm_filtered_NT_CDS")
tree_dir = os.path.join(W_DIR, "OrthoMamv12_nogaps/trees")
res_dir = os.path.join(W_DIR, "OrthoMamv12_nogaps/sims")
MODEL_PATH = "/Users/dsilvestro/Documents/Projects/Ongoing/GenAli/transformer_res/axial_msa_transformer_results/"
TRAINED_MODEL = "axial_msa_transformer.pth"
model = AxialMSATransformer(
    in_channels=len(CHANNEL_DICT), latent_dim=LATENT_DIM, max_sites=MAX_SITES
)
device = torch.device("cpu")
model.load_state_dict(
    torch.load(os.path.join(MODEL_PATH, TRAINED_MODEL), map_location=device)
)
model.to(device)

simulate_data_alisim_from_tree(
    msa_dir=msa_dir,
    tree_dir=tree_dir,
    res_dir=res_dir,
    bin_dir=bin_dir,
    model_options=["anc_seq"],
    evol_model="GTR",
    evol_model_tag="GTR",
    n_sims=1000,
    seed=42,
)  #

simulate_data_alisim_from_tree(
    msa_dir=msa_dir,
    tree_dir=tree_dir,
    res_dir=res_dir,
    bin_dir=bin_dir,
    model_options=["anc_seq"],
    evol_model_tag="GTR_Gamma",
    n_sims=1000,
    seed=42,
)  #

simulate_data_alisim_from_tree(
    msa_dir=msa_dir,
    tree_dir=tree_dir,
    res_dir=res_dir,
    bin_dir=bin_dir,
    model_options=["anc_seq"],
    evol_model="JC",
    evol_model_tag="JC",
    n_sims=1000,
    seed=42,
)  #

simulate_data_alisim_from_tree(
    msa_dir=msa_dir,
    tree_dir=tree_dir,
    res_dir=res_dir,
    bin_dir=bin_dir,
    model_options=["anc_seq", "CODON"],
    evol_model="ECMK07+F+R12",
    evol_model_tag="codon",
    n_sims=1000,
    seed=42,
)  #

# 1. Extract embeddings

sim_files = np.sort(
    glob.glob(os.path.join(W_DIR, f"OrthoMamv12_nogaps/sims/GenAli_20260619/*"))
)[:1000]
all_embeddings_sim = []
CHANNEL_DICT = ["A", "C", "G", "T"]

model.eval()
with torch.no_grad():
    i = 0
    for f_path in sim_files:
        try:
            data_np = parse_file(f_path, schema="phylip", channel_dict=CHANNEL_DICT)
        except:
            data_np = parse_file(f_path, channel_dict=CHANNEL_DICT)

        # Trim sites if too long
        if data_np.shape[2] > MAX_SITES:
            start = np.random.randint(0, data_np.shape[-1] - MAX_SITES)
            data_np = data_np[:, :, start: start + MAX_SITES]

        # Sub-sample taxa
        if data_np.shape[1] > MAX_TAXA:
            indices = np.random.choice(data_np.shape[1], MAX_TAXA, replace=False)
            data_np = data_np[:, indices, :]

        data = torch.from_numpy(data_np).float().unsqueeze(0)  # .to(device)
        latent = model.encode(data)
        all_embeddings_sim.append(latent.squeeze().cpu().numpy())
        print_update(
            f"Processed: {os.path.basename(f_path)} ({i + 1} / {len(sim_files)})"
        )
        i += 1

# Convert to a 2D numpy array [num_samples, latent_dim]
data_sim = np.array(all_embeddings_sim)

# Save sim embeddings
data_to_save = {
    "file_name": [os.path.basename(f) for f in sim_files[: len(data_sim)]],
}

# 2. Add the embedding dimensions (e.g., dim_0, dim_1, ...)
for i in range(data_sim.shape[1]):
    data_to_save[f"dim_{i}"] = data_sim[:, i]

# 3. Create DataFrame and save
os.makedirs(res_dir, exist_ok=True)
df = pd.DataFrame(data_to_save)
sub_model = os.path.dirname(sim_files[0]).split("/")[-1]
df.to_csv(os.path.join(res_dir, f"embeddings_{sub_model}.csv"), index=False)
print(f"Saved embeddings to embeddings_{sub_model}.csv")

# RUN UMAP
# 1. Load Real Data
f = os.path.join(MODEL_PATH, "embeddings_results.csv")
data = pd.read_csv(f)
# 2. Extract the 128 latent variables
latent_cols = [f"dim_{i}" for i in range(128)]
X_train = data[latent_cols].values

# 3. Initialize and FIT the UMAP reducer on the training set
# We keep the reducer object to transform the test set later
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=123)
embedding_train = reducer.fit_transform(X_train)

# Update the dataframe with new UMAP coordinates if desired
data["umap_0"] = embedding_train[:, 0]
data["umap_1"] = embedding_train[:, 1]

for sub_model in [
    "alisim_anc_seqJC",
    "alisim_anc_seqGTR",
    "alisim_anc_seqGTR_Gamma",
    "alisim_anc_seq_codon",
    "alisim_GTR_G4_empirical_prm",
    "GenAli_20260619",
]:
    # sim data set
    f = os.path.join(res_dir, f"embeddings_{sub_model}.csv")
    data_sim = pd.read_csv(f)
    latent_cols = [f"dim_{i}" for i in range(128)]
    X_sim = data_sim[latent_cols].values

    # UMAP: using .transform() to project into the same space
    embedding_sim = reducer.transform(X_sim)
    data_sim["umap_0"] = embedding_sim[:, 0]
    data_sim["umap_1"] = embedding_sim[:, 1]

    # Subset the dataframe (to re-plot
    # target_files = [os.path.basename(f).split("_sim")[0] + ".fasta" for f in sim_files]
    # subset_df = data[data["file_name"].isin(target_files)]
    #
    # fig, axes = plt.subplots(2, 5, figsize=(25, 10))

    # Save the plot
    plt.figure(figsize=(8, 7))
    plt.scatter(
        data["umap_0"],
        data["umap_1"],
        s=12,
        c="#cccccc",  # "#6baed6",
        label="All alignments",
        alpha=0.6,
    )
    # plt.scatter(
    #     subset_df["umap_0"],
    #     subset_df["umap_1"],
    #     s=12,
    #     color="#08519c",
    #     label="Selected alignments",
    #     alpha=0.8,
    # )
    plt.scatter(
        data_sim["umap_0"],
        data_sim["umap_1"],
        s=12,
        c="#d95f02",
        label="Simulated alignments",
        alpha=0.8,
    )
    plt.legend(loc="best", markerscale=2.0, frameon=True, fontsize="small")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(f"Empirical UMAP Embedding: {sub_model}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(res_dir, f"umap_sim_projection_{sub_model}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

##############################################################################
#                2D histogram
##############################################################################
# 1. Load Real Data
f = os.path.join(MODEL_PATH, "embeddings_results.csv")
data = pd.read_csv(f)
# 2. Extract the 128 latent variables
latent_cols = [f"dim_{i}" for i in range(128)]
X_train = data[latent_cols].values

# 3. Initialize and FIT the UMAP reducer on the training set
# We keep the reducer object to transform the test set later
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=123)
embedding_train = reducer.fit_transform(X_train)

# test set
data_sim = pd.read_csv(
    os.path.join(W_DIR, "transformer_res/axial_msa_transformer_results/embeddings_results_test.csv")
)
latent_cols = [f"dim_{i}" for i in range(128)]
X_test = data_sim[latent_cols].values
embedding_test = reducer.transform(X_test)

js_distance_null_distribution = []

for sub_model in [
    "alisim_anc_seqJC",
    "alisim_anc_seqGTR",
    "alisim_anc_seqGTR_Gamma",
    "alisim_anc_seq_codon",
    "alisim_GTR_G4_empirical_prm",
    "GenAli_20260619",
    "results_test_set1000",
]:
    # load simulations
    try:
        model_name = sub_model.split("seq")[1].replace("_", " ")
    except:
        model_name = sub_model
    f = os.path.join(res_dir, f"embeddings_{sub_model}.csv")
    data_sim = pd.read_csv(f)
    latent_cols = [f"dim_{i}" for i in range(128)]
    X_sim = data_sim[latent_cols].values
    embedding_sim = reducer.transform(X_sim)

    # Create the 2D histogram plot
    plt.figure(figsize=(24, 5))

    plt.subplot(1, 4, 1)
    # Capture the edges here:
    counts1, xedges, yedges, im1 = plt.hist2d(
        embedding_train[:, 0], embedding_train[:, 1], bins=40, cmap="Purples"
    )
    plt.colorbar(im1, label="Counts")
    plt.title("Empirical distribution")
    plt.subplot(1, 4, 2)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    # --- 3. PLOT SECOND HISTOGRAM USING CAPTURED EDGES ---
    plt.subplot(1, 4, 2)
    # Pass a tuple of (xedges, yedges) into the bins parameter:
    counts2, _, _, im2 = plt.hist2d(
        embedding_sim[:, 0], embedding_sim[:, 1], bins=(xedges, yedges), cmap="Purples"
    )
    plt.colorbar(im2, label="Counts")
    plt.title(f"Simulated distribution ({model_name})")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    # --- 1. Recreate the bins and counts from the previous step ---
    # (Assuming counts1, counts2, xedges, yedges are captured from plt.hist2d)
    # For demonstration, let's normalize raw counts to probability distributions:
    P = counts1 / np.sum(counts1)  # Grid 1 sums to 1.0
    Q = counts2 / np.sum(counts2)  # Grid 2 sums to 1.0

    # --- 2. Calculate the Difference Grid ---
    # Positive values = Dataset 2 has higher density
    # Negative values = Dataset 1 has higher density
    density_diff = Q - P

    # --- 3. Plot the Difference Heatmap ---
    plt.subplot(1, 4, 3)

    # Find the maximum absolute value to center the colorbar perfectly at 0
    max_val = 0.015  # np.max(np.abs(density_diff))

    # Use pcolormesh to plot using the explicit edge coordinates
    im = plt.pcolormesh(
        xedges, yedges, density_diff.T, cmap="bwr", vmin=-max_val, vmax=max_val
    )

    # Add elements
    cbar = plt.colorbar(im, label="Density Difference (Dataset 2 - Dataset 1)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("2D Density Difference Map")

    # Add a text box to display the distance metric
    # (We will calculate js_distance in Part 2 below)
    js_distance_obs = jensenshannon(P.flatten(), Q.flatten())
    print("Jensen-Shannon distance: ", js_distance_obs)

    # get expected jensen-shannon distance

    if len(js_distance_null_distribution) == 0:
        for _ in range(1000):
            rnd_indx = np.random.permutation(len(embedding_test))[:1000]
            counts2, _, _, im2 = plt.hist2d(
                embedding_test[rnd_indx, 0],
                embedding_test[rnd_indx, 1],
                bins=(xedges, yedges),
                cmap="viridis",
            )
            P = counts1 / np.sum(counts1)  # Grid 1 sums to 1.0
            Q = counts2 / np.sum(counts2)  # Grid 2 sums to 1.0
            js_distance_null_distribution.append(
                jensenshannon(P.flatten(), Q.flatten())
            )

    np.quantile(js_distance_null_distribution, q=[0.025, 0.5, 0.975])

    # Plot the smooth density kernel
    plt.subplot(1, 4, 4)
    sns.kdeplot(
        js_distance_null_distribution,
        fill=True,
        color="#3182bd",
        alpha=0.4,
        linewidth=2,
        label="Null Distribution ($H_0$)",
    )

    # Add the vertical line for the observed statistic
    plt.axvline(
        x=js_distance_obs,
        color="#de2d26",
        linestyle="--",
        linewidth=2.5,
        label=f"Observed Distance ({js_distance_obs:.4f})",
    )

    plt.xlim(0.4, 0.6)

    # Formatting and clean aesthetic
    plt.title("Permutation Test: Observed vs. Null JS Distance", fontsize=12, pad=12)
    plt.xlabel("Jensen-Shannon Distance", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(
        os.path.join(res_dir, f"delta_{model_name}.pdf"), dpi=300, bbox_inches="tight"
    )
    # plt.show()
