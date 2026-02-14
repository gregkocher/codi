# ABOUTME: Shared analysis and visualization utilities for CODI experiments.
# ABOUTME: Provides compute functions (cosine similarity, drift, etc.) and
# ABOUTME: visualization functions (heatmaps, t-SNE, UMAP, cross-layer analysis)
# ABOUTME: with configurable axis labels via x_display_name parameter.

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from sklearn.manifold import TSNE


# ============================================================================
# Compute / Analysis Functions
# ============================================================================


def compute_cosine_similarity_matrices(all_vectors, x_values):
    """
    For each latent position, compute a pairwise cosine similarity matrix
    across all keys in x_values.

    Args:
        all_vectors: dict mapping key -> list of (1,1,hidden_dim) tensors
        x_values: list of keys (e.g. X values or rollout indices)

    Returns:
        dict mapping position_index -> (N x N) numpy array of cosine similarities
    """
    num_positions = len(all_vectors[x_values[0]])
    cos_matrices = {}

    for pos in range(num_positions):
        vecs = []
        for x in x_values:
            v = all_vectors[x][pos].flatten().float()
            vecs.append(v)
        vecs = torch.stack(vecs)  # (N, hidden_dim)

        norms = vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        vecs_normed = vecs / norms
        cos_mat = (vecs_normed @ vecs_normed.T).numpy()
        cos_matrices[pos] = cos_mat

    return cos_matrices


def compute_consecutive_cosine_sims(all_vectors, x_values):
    """
    For each latent position, compute cosine similarity between consecutive
    keys: sim(k, k+1). Tests monotonicity/smoothness.

    Returns dict mapping position -> list of (N-1) cosine similarities.
    """
    num_positions = len(all_vectors[x_values[0]])
    consec_sims = {}

    for pos in range(num_positions):
        sims = []
        for i in range(len(x_values) - 1):
            v1 = all_vectors[x_values[i]][pos].flatten().float()
            v2 = all_vectors[x_values[i + 1]][pos].flatten().float()
            sim = torch.nn.functional.cosine_similarity(v1, v2, dim=0).item()
            sims.append(sim)
        consec_sims[pos] = sims

    return consec_sims


def compute_drift_from_first(all_vectors, x_values):
    """
    For each latent position, compute cosine similarity between the first
    key's vector and every other key's vector.

    Returns dict mapping position -> list of N cosine similarities.
    """
    num_positions = len(all_vectors[x_values[0]])
    drift = {}

    for pos in range(num_positions):
        v_first = all_vectors[x_values[0]][pos].flatten().float()
        sims = []
        for x in x_values:
            v = all_vectors[x][pos].flatten().float()
            sim = torch.nn.functional.cosine_similarity(v_first, v, dim=0).item()
            sims.append(sim)
        drift[pos] = sims

    return drift


# ============================================================================
# Internal Helpers for Dimensionality Reduction Plots
# ============================================================================


def _flatten_vectors(all_vectors, x_values):
    """Flatten all latent vectors into arrays + label arrays for plotting."""
    num_positions = len(all_vectors[x_values[0]])
    all_vecs = []
    labels_position = []
    labels_x = []
    for x in x_values:
        for pos in range(num_positions):
            v = all_vectors[x][pos].flatten().float().numpy()
            all_vecs.append(v)
            labels_position.append(pos)
            labels_x.append(x)
    return (
        np.array(all_vecs),
        np.array(labels_position),
        np.array(labels_x),
        num_positions,
    )


def _x_color_norm(x_values):
    """Return a color normalizer for values; uses LogNorm when range > 100x."""
    x_min, x_max = min(x_values), max(x_values)
    if x_min > 0 and x_max / x_min > 100:
        return mcolors.LogNorm(vmin=x_min, vmax=x_max)
    return plt.Normalize(vmin=x_min, vmax=x_max)


def _plot_embeddings_by_position(
    embeddings, labels_position, x_values, num_positions, ax, method_name,
    x_display_name="X",
):
    """Plot 2-D embeddings colored by latent position with connecting lines."""
    cmap = plt.cm.tab10
    for pos in range(num_positions):
        mask = labels_position == pos
        idxs = np.where(mask)[0]
        color = cmap(pos % 10)
        label = f"Latent {pos}"

        ax.plot(
            embeddings[idxs, 0], embeddings[idxs, 1],
            "-", color=color, linewidth=1.0, alpha=0.4, zorder=1,
        )
        ax.scatter(
            embeddings[idxs, 0], embeddings[idxs, 1],
            c=[color], s=50, label=label, zorder=2, edgecolors="white",
            linewidth=0.5,
        )
        ax.annotate(
            f"{x_display_name}={x_values[0]}",
            (embeddings[idxs[0], 0], embeddings[idxs[0], 1]),
            fontsize=6, color=color, alpha=0.8,
        )
        ax.annotate(
            f"{x_display_name}={x_values[-1]}",
            (embeddings[idxs[-1], 0], embeddings[idxs[-1], 1]),
            fontsize=6, color=color, alpha=0.8,
        )

    ax.set_xlabel(f"{method_name} dim 1", fontsize=12)
    ax.set_ylabel(f"{method_name} dim 2", fontsize=12)
    ax.set_title(
        f"{method_name} of Latent Vectors "
        f"(colored by position, connected across {x_display_name})",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.2)


def _plot_embeddings_by_x(
    embeddings, labels_x, x_values, ax, method_name,
    x_display_name="X",
):
    """Plot 2-D embeddings colored by key value with connecting lines."""
    norm = _x_color_norm(x_values)
    cmap_x = plt.cm.coolwarm

    for x in x_values:
        mask = labels_x == x
        idxs = np.where(mask)[0]
        color = cmap_x(norm(x))

        ax.plot(
            embeddings[idxs, 0], embeddings[idxs, 1],
            "-", color=color, linewidth=1.0, alpha=0.4, zorder=1,
        )
        ax.scatter(
            embeddings[idxs, 0], embeddings[idxs, 1],
            c=[color], s=50, zorder=2, edgecolors="white", linewidth=0.5,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap_x, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(x_display_name, fontsize=12)

    ax.set_xlabel(f"{method_name} dim 1", fontsize=12)
    ax.set_ylabel(f"{method_name} dim 2", fontsize=12)
    ax.set_title(
        f"{method_name} of Latent Vectors "
        f"(colored by {x_display_name}, connected across positions)",
        fontsize=14, fontweight="bold",
    )
    ax.grid(True, alpha=0.2)


# ============================================================================
# Visualization Functions
# ============================================================================


def visualize_cosine_similarity_matrices(
    cos_matrices, x_values, results_dir, x_display_name="X",
):
    """One heatmap per latent position showing pairwise cosine similarity."""
    num_positions = len(cos_matrices)
    cols = min(num_positions, 4)
    rows = (num_positions + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5), squeeze=False)

    for pos in range(num_positions):
        ax = axes[pos // cols][pos % cols]
        mat = cos_matrices[pos]

        im = ax.imshow(mat, cmap="RdYlGn", aspect="equal", vmin=0.9, vmax=1.0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        label = f"Latent {pos}"
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(x_display_name, fontsize=9)
        ax.set_ylabel(x_display_name, fontsize=9)

        all_ticks = list(range(len(x_values)))
        all_labels = [str(x) for x in x_values]
        ax.set_xticks(all_ticks)
        ax.set_xticklabels(all_labels, fontsize=6, rotation=90, ha="center")
        ax.set_yticks(all_ticks)
        ax.set_yticklabels(all_labels, fontsize=6)

    for idx in range(num_positions, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    plt.suptitle(
        f"Pairwise Cosine Similarity of Latent Vectors Across {x_display_name}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    path = results_dir / "cosine_similarity_matrices.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_drift_from_first(
    drift, x_values, results_dir, x_display_name="X",
):
    """
    Line plot: for each latent position, how does cosine similarity to the
    first key's vector change? Tests monotonic drift.
    """
    indices = list(range(len(x_values)))
    x_labels = [str(x) for x in x_values]

    fig, ax = plt.subplots(figsize=(max(12, len(x_values) * 0.5), 5))

    cmap = plt.cm.coolwarm
    num_positions = len(drift)
    colors = [cmap(i / max(1, num_positions - 1)) for i in range(num_positions)]

    for pos in range(num_positions):
        label = f"Latent {pos}"
        ax.plot(
            indices, drift[pos], "o-",
            color=colors[pos], markersize=5, linewidth=1.5,
            label=label,
        )

    ax.set_xlabel(x_display_name, fontsize=13, fontweight="bold")
    ax.set_ylabel(
        f"Cosine similarity to {x_display_name}={x_values[0]}",
        fontsize=13, fontweight="bold",
    )
    ax.set_title(
        f"Latent Vector Drift from First {x_display_name}",
        fontsize=15, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(indices)
    ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")

    plt.tight_layout()
    path = results_dir / "drift_from_first.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_consecutive_cosine_sims(
    consec_sims, x_values, results_dir, x_display_name="X",
):
    """
    Line plot: cosine similarity between consecutive keys for each
    latent position. Flat = smooth latent space, dips = phase transitions.
    """
    midpoint_labels = [
        f"{x_values[i]}-{x_values[i + 1]}" for i in range(len(x_values) - 1)
    ]
    indices = list(range(len(midpoint_labels)))

    fig, ax = plt.subplots(figsize=(max(12, len(indices) * 0.5), 5))

    cmap = plt.cm.coolwarm
    num_positions = len(consec_sims)
    colors = [cmap(i / max(1, num_positions - 1)) for i in range(num_positions)]

    for pos in range(num_positions):
        label = f"Latent {pos}"
        ax.plot(
            indices, consec_sims[pos], "o-",
            color=colors[pos], markersize=5, linewidth=1.5,
            label=label,
        )

    ax.set_xlabel(
        f"Consecutive {x_display_name} pair", fontsize=13, fontweight="bold",
    )
    ax.set_ylabel(
        f"Cosine similarity ({x_display_name}_i, {x_display_name}_{{i+1}})",
        fontsize=13, fontweight="bold",
    )
    ax.set_title(
        f"Consecutive Cosine Similarity Between Adjacent {x_display_name}s",
        fontsize=15, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(indices)
    ax.set_xticklabels(midpoint_labels, fontsize=6, rotation=45, ha="right")

    plt.tight_layout()
    path = results_dir / "consecutive_cosine_similarity.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_vector_norms(
    all_vectors, x_values, results_dir, x_display_name="X",
):
    """
    Plot L2 norms of latent vectors across keys for each position.
    Checks whether vector magnitude changes systematically.
    """
    indices = list(range(len(x_values)))
    x_labels = [str(x) for x in x_values]
    num_positions = len(all_vectors[x_values[0]])

    fig, ax = plt.subplots(figsize=(max(12, len(x_values) * 0.5), 5))

    cmap = plt.cm.coolwarm
    colors = [cmap(i / max(1, num_positions - 1)) for i in range(num_positions)]

    for pos in range(num_positions):
        norms = []
        for x in x_values:
            v = all_vectors[x][pos].flatten().float()
            norms.append(v.norm().item())
        label = f"Latent {pos}"
        ax.plot(
            indices, norms, "o-",
            color=colors[pos], markersize=5, linewidth=1.5,
            label=label,
        )

    ax.set_xlabel(x_display_name, fontsize=13, fontweight="bold")
    ax.set_ylabel("L2 Norm", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Latent Vector L2 Norms vs {x_display_name}",
        fontsize=15, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(indices)
    ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")

    plt.tight_layout()
    path = results_dir / "vector_norms.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_tsne(
    all_vectors, x_values, results_dir,
    x_display_name="X", perplexity=5, seed=42,
):
    """
    t-SNE plot of all latent vectors. Two sub-plots: colored by latent
    position and colored by key value.
    """
    all_vecs, labels_position, labels_x, num_positions = _flatten_vectors(
        all_vectors, x_values
    )

    effective_perplexity = min(perplexity, max(2, len(all_vecs) // 4))
    tsne = TSNE(
        n_components=2, perplexity=effective_perplexity,
        random_state=seed, max_iter=2000, learning_rate="auto", init="pca",
    )
    embeddings = tsne.fit_transform(all_vecs)

    # Plot 1: colored by latent position
    fig, ax = plt.subplots(figsize=(12, 9))
    _plot_embeddings_by_position(
        embeddings, labels_position, x_values, num_positions, ax, "t-SNE",
        x_display_name=x_display_name,
    )
    plt.tight_layout()
    path = results_dir / "tsne_by_position.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")

    # Plot 2: colored by key value
    fig, ax = plt.subplots(figsize=(12, 9))
    _plot_embeddings_by_x(
        embeddings, labels_x, x_values, ax, "t-SNE",
        x_display_name=x_display_name,
    )
    plt.tight_layout()
    path = results_dir / "tsne_by_x_value.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_umap(
    all_vectors, x_values, results_dir,
    x_display_name="X", n_neighbors=15, min_dist=0.1, seed=42,
):
    """
    UMAP plot of all latent vectors. Same two sub-plots as t-SNE:
    colored by latent position and colored by key value.
    """
    all_vecs, labels_position, labels_x, num_positions = _flatten_vectors(
        all_vectors, x_values
    )

    effective_neighbors = min(n_neighbors, max(2, len(all_vecs) - 1))
    reducer = umap.UMAP(
        n_components=2, n_neighbors=effective_neighbors,
        min_dist=min_dist, random_state=seed, metric="cosine",
    )
    embeddings = reducer.fit_transform(all_vecs)

    # Plot 1: colored by latent position
    fig, ax = plt.subplots(figsize=(12, 9))
    _plot_embeddings_by_position(
        embeddings, labels_position, x_values, num_positions, ax, "UMAP",
        x_display_name=x_display_name,
    )
    plt.tight_layout()
    path = results_dir / "umap_by_position.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")

    # Plot 2: colored by key value
    fig, ax = plt.subplots(figsize=(12, 9))
    _plot_embeddings_by_x(
        embeddings, labels_x, x_values, ax, "UMAP",
        x_display_name=x_display_name,
    )
    plt.tight_layout()
    path = results_dir / "umap_by_x_value.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================================
# Logit Lens Heatmaps
# ============================================================================


def visualize_logit_lens_heatmap(
    all_results, x_values, tokenizer, results_dir,
    x_display_name="X", layer_index=None,
):
    """
    Heatmap: rows = keys (X values or rollout indices), columns = latent
    vector index (0..K). Each cell shows top-1 token from a specific layer's
    logit lens.

    Args:
        all_results: dict mapping key -> result dict with 'latent_logit_lens'
        x_values: list of keys
        x_display_name: Label for the y-axis / titles (e.g. "X" or "Rollout")
        layer_index: Index into the logit_lens list for each position.
            None (default) = last layer (-1).
    """
    num_x = len(x_values)
    num_positions = len(all_results[x_values[0]]["latent_logit_lens"])

    li = -1 if layer_index is None else layer_index

    token_matrix = []
    prob_matrix = np.zeros((num_x, num_positions))

    for row_idx, x in enumerate(x_values):
        row_tokens = []
        for p_idx, pos_data in enumerate(all_results[x]["latent_logit_lens"]):
            layer_data = pos_data["logit_lens"][li]
            top_token_id = layer_data["top_indices"][0]
            top_prob = layer_data["top_probs"][0]
            token_str = tokenizer.decode([top_token_id])
            row_tokens.append(token_str)
            prob_matrix[row_idx, p_idx] = top_prob
        token_matrix.append(row_tokens)

    fig, ax = plt.subplots(
        figsize=(3 + num_positions * 1.8, 2 + num_x * 0.55)
    )

    im = ax.imshow(prob_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Top-1 probability", rotation=270, labelpad=15, fontsize=12)

    for i in range(num_x):
        for j in range(num_positions):
            token = token_matrix[i][j]
            prob = prob_matrix[i, j]
            token_display = repr(token)[1:-1] if token else ""
            text_color = "white" if prob > 0.5 else "black"
            ax.text(
                j, i, token_display,
                ha="center", va="center", color=text_color, fontsize=8,
            )

    # Resolve actual layer number for title/filename
    actual_layer = (
        all_results[x_values[0]]["latent_logit_lens"][0]["logit_lens"][li]["layer"]
    )

    ax.set_xlabel("Latent Vector Index", fontsize=13, fontweight="bold")
    ax.set_ylabel(x_display_name, fontsize=13, fontweight="bold")
    ax.set_xticks(range(num_positions))
    ax.set_xticklabels([str(i) for i in range(num_positions)], fontsize=10)
    ax.set_yticks(range(num_x))
    ax.set_yticklabels([str(x) for x in x_values], fontsize=10)

    if layer_index is None:
        title = f"Top-1 Logit Lens Token (Final Layer) vs {x_display_name}"
        filename = "logit_lens_heatmap.png"
    else:
        title = f"Top-1 Logit Lens Token (Layer {actual_layer}) vs {x_display_name}"
        filename = f"logit_lens_heatmap_layer_{actual_layer}.png"

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    path = results_dir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_logit_lens_heatmap_top5(
    all_results, x_values, tokenizer, results_dir,
    x_display_name="X", layer_index=None,
):
    """
    Heatmap: rows = keys, columns = latent vector index (0..K).
    Each cell shows the top-1 token prominently and tokens 2-5 in smaller
    font below in parentheses.

    Args:
        x_display_name: Label for the y-axis / titles (e.g. "X" or "Rollout")
        layer_index: Index into the logit_lens list for each position.
            None (default) = last layer (-1).
    """
    num_x = len(x_values)
    num_positions = len(all_results[x_values[0]]["latent_logit_lens"])

    li = -1 if layer_index is None else layer_index

    top5_matrix = []  # [row][col] = list of 5 token strings
    prob_matrix = np.zeros((num_x, num_positions))

    for row_idx, x in enumerate(x_values):
        row_top5 = []
        for p_idx, pos_data in enumerate(all_results[x]["latent_logit_lens"]):
            layer_data = pos_data["logit_lens"][li]
            tokens_5 = []
            for k in range(min(5, len(layer_data["top_indices"]))):
                tid = layer_data["top_indices"][k]
                tokens_5.append(tokenizer.decode([tid]))
            row_top5.append(tokens_5)
            prob_matrix[row_idx, p_idx] = layer_data["top_probs"][0]
        top5_matrix.append(row_top5)

    # Resolve actual layer number for title/filename
    actual_layer = (
        all_results[x_values[0]]["latent_logit_lens"][0]["logit_lens"][li]["layer"]
    )

    fig, ax = plt.subplots(
        figsize=(3 + num_positions * 2.2, 2 + num_x * 0.85)
    )

    im = ax.imshow(prob_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Top-1 probability", rotation=270, labelpad=15, fontsize=12)

    for i in range(num_x):
        for j in range(num_positions):
            tokens = top5_matrix[i][j]
            prob = prob_matrix[i, j]
            text_color = "white" if prob > 0.5 else "black"

            # Top-1 token — prominent
            top1_display = repr(tokens[0])[1:-1] if tokens else ""
            ax.text(
                j, i - 0.15, top1_display,
                ha="center", va="center", color=text_color,
                fontsize=8, fontweight="bold",
            )

            # Tokens 2-5 — smaller, in parentheses below
            if len(tokens) > 1:
                rest = ", ".join(repr(t)[1:-1] for t in tokens[1:5])
                ax.text(
                    j, i + 0.2, f"({rest})",
                    ha="center", va="center", color=text_color,
                    fontsize=5,
                )

    ax.set_xlabel("Latent Vector Index", fontsize=13, fontweight="bold")
    ax.set_ylabel(x_display_name, fontsize=13, fontweight="bold")
    ax.set_xticks(range(num_positions))
    ax.set_xticklabels([str(i) for i in range(num_positions)], fontsize=10)
    ax.set_yticks(range(num_x))
    ax.set_yticklabels([str(x) for x in x_values], fontsize=10)

    if layer_index is None:
        title = f"Top-5 Logit Lens Tokens (Final Layer) vs {x_display_name}"
        filename = "logit_lens_heatmap_top5.png"
    else:
        title = f"Top-5 Logit Lens Tokens (Layer {actual_layer}) vs {x_display_name}"
        filename = f"logit_lens_heatmap_top5_layer_{actual_layer}.png"

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    path = results_dir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_full_logit_lens_grid(
    all_results, x_values, tokenizer, results_dir,
    x_display_name="X",
):
    """
    Multi-panel figure: one heatmap per key (rollout or X value) showing
    (analyzed_layer x latent_position) with top-1 token annotations.

    Args:
        all_results: dict mapping key -> result dict with 'latent_logit_lens'
        x_values: list of keys
        x_display_name: Label for subplot titles (e.g. "X" or "Rollout")
    """
    num_items = len(x_values)
    num_positions = len(all_results[x_values[0]]["latent_logit_lens"])

    # Determine the set of analyzed layer indices from the first item
    analyzed_layers = [
        entry["layer"]
        for entry in all_results[x_values[0]]["latent_logit_lens"][0]["logit_lens"]
    ]
    num_analyzed = len(analyzed_layers)

    cols = min(num_items, 5)
    rows = (num_items + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 5, rows * 4),
        squeeze=False,
    )

    for item_idx, key in enumerate(x_values):
        ax = axes[item_idx // cols][item_idx % cols]
        result = all_results[key]

        prob_matrix = np.zeros((num_analyzed, num_positions))
        token_matrix = [[None] * num_positions for _ in range(num_analyzed)]

        for p_idx, pos_data in enumerate(result["latent_logit_lens"]):
            for row_idx, layer_data in enumerate(pos_data["logit_lens"]):
                top_token_id = layer_data["top_indices"][0]
                top_prob = layer_data["top_probs"][0]
                prob_matrix[row_idx, p_idx] = top_prob
                token_matrix[row_idx][p_idx] = tokenizer.decode([top_token_id])

        ax.imshow(prob_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

        # Only annotate if small enough to be readable
        if num_analyzed <= 20 and num_positions <= 10:
            for i in range(num_analyzed):
                for j in range(num_positions):
                    token = token_matrix[i][j]
                    prob = prob_matrix[i, j]
                    token_display = repr(token)[1:-1] if token else ""
                    text_color = "white" if prob > 0.5 else "black"
                    ax.text(
                        j, i, token_display,
                        ha="center", va="center", color=text_color, fontsize=7,
                    )

        correct_str = "correct" if result.get("correct", False) else "wrong"
        text_preview = result.get("generated_text", "")[:20]
        ax.set_title(
            f"{x_display_name} {key} ({correct_str}: {text_preview})",
            fontsize=10,
        )
        ax.set_xlabel("Latent Vector Index", fontsize=9)
        ax.set_ylabel("Layer", fontsize=9)
        ax.set_xticks(range(num_positions))
        ax.set_xticklabels([str(i) for i in range(num_positions)], fontsize=8)
        ax.set_yticks(range(num_analyzed))
        ax.set_yticklabels([str(l) for l in analyzed_layers], fontsize=8)

    # Hide unused axes
    for idx in range(num_items, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    plt.tight_layout()
    output_path = results_dir / "logit_lens_full_grid.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# Cross-Layer Analysis
# ============================================================================


def visualize_token_first_occurrence_depth(
    all_results, x_values, tokenizer, results_dir,
    x_display_name="X",
):
    """
    Heatmap showing the earliest layer at which the top-1 token matches the
    final layer's top-1 token and never changes again (first occurrence depth).
    Rows = keys, columns = latent vector index.
    """
    num_x = len(x_values)
    num_positions = len(all_results[x_values[0]]["latent_logit_lens"])

    depth_matrix = np.zeros((num_x, num_positions))

    for row_idx, x in enumerate(x_values):
        for p_idx, pos_data in enumerate(all_results[x]["latent_logit_lens"]):
            layers_data = pos_data["logit_lens"]
            num_layers = len(layers_data)
            final_token = layers_data[-1]["top_indices"][0]

            # Scan backwards to find the earliest layer that matches the final
            # token and all subsequent layers also match.
            first_layer = num_layers - 1  # worst case: only final layer
            for li in range(num_layers - 1, -1, -1):
                if layers_data[li]["top_indices"][0] == final_token:
                    first_layer = li
                else:
                    break  # the streak is broken

            depth_matrix[row_idx, p_idx] = layers_data[first_layer]["layer"]

    fig, ax = plt.subplots(
        figsize=(3 + num_positions * 1.8, 2 + num_x * 0.55)
    )

    im = ax.imshow(depth_matrix, cmap="coolwarm", aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("First occurrence layer", rotation=270, labelpad=15, fontsize=12)

    for i in range(num_x):
        for j in range(num_positions):
            val = int(depth_matrix[i, j])
            text_color = "black"
            ax.text(
                j, i, str(val),
                ha="center", va="center", color=text_color, fontsize=8,
            )

    ax.set_xlabel("Latent Vector Index", fontsize=13, fontweight="bold")
    ax.set_ylabel(x_display_name, fontsize=13, fontweight="bold")
    ax.set_xticks(range(num_positions))
    ax.set_xticklabels([str(i) for i in range(num_positions)], fontsize=10)
    ax.set_yticks(range(num_x))
    ax.set_yticklabels([str(x) for x in x_values], fontsize=10)
    ax.set_title(
        "Token 1st Occurrence Depth (earliest layer matching final top-1 token)",
        fontsize=13, fontweight="bold", pad=12,
    )

    plt.tight_layout()
    path = results_dir / "token_first_occurrence_depth.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_layer_entropy(
    all_results, x_values, results_dir, x_display_name="X",
):
    """
    Heatmap: rows = layers, columns = latent vector index.
    Each cell = mean entropy (over top-K probs) across all keys.
    """
    num_positions = len(all_results[x_values[0]]["latent_logit_lens"])
    sample_lens = all_results[x_values[0]]["latent_logit_lens"][0]["logit_lens"]
    num_layers = len(sample_lens)
    layer_numbers = [ld["layer"] for ld in sample_lens]

    # Accumulate entropy per (layer, position)
    entropy_sums = np.zeros((num_layers, num_positions))

    for x in x_values:
        for p_idx, pos_data in enumerate(all_results[x]["latent_logit_lens"]):
            for li, layer_data in enumerate(pos_data["logit_lens"]):
                probs = np.array(layer_data["top_probs"], dtype=np.float64)
                probs = probs[probs > 0]  # avoid log(0)
                entropy = -np.sum(probs * np.log2(probs))
                entropy_sums[li, p_idx] += entropy

    entropy_mean = entropy_sums / len(x_values)

    fig, ax = plt.subplots(
        figsize=(3 + num_positions * 1.8, 2 + num_layers * 0.45)
    )

    im = ax.imshow(entropy_mean, cmap="coolwarm", aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean entropy (bits, top-K)", rotation=270, labelpad=15, fontsize=11)

    ax.set_xlabel("Latent Vector Index", fontsize=13, fontweight="bold")
    ax.set_ylabel("Layer", fontsize=13, fontweight="bold")
    ax.set_xticks(range(num_positions))
    ax.set_xticklabels([str(i) for i in range(num_positions)], fontsize=10)
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([str(l) for l in layer_numbers], fontsize=9)
    ax.set_title(
        f"Layer-wise Mean Entropy of Logit Lens (across {x_display_name} values)",
        fontsize=13, fontweight="bold", pad=12,
    )

    plt.tight_layout()
    path = results_dir / "layer_entropy_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_token_stability_across_layers(
    all_results, x_values, results_dir, x_display_name="X",
):
    """
    Line chart: for each latent position, what fraction of keys have the
    same top-1 token as the final layer, at each layer depth?
    """
    num_positions = len(all_results[x_values[0]]["latent_logit_lens"])
    sample_lens = all_results[x_values[0]]["latent_logit_lens"][0]["logit_lens"]
    num_layers = len(sample_lens)
    layer_numbers = [ld["layer"] for ld in sample_lens]
    num_x = len(x_values)

    # stability_matrix[li][pos] = fraction of keys matching final top-1
    stability = np.zeros((num_layers, num_positions))

    for p_idx in range(num_positions):
        for li in range(num_layers):
            match_count = 0
            for x in x_values:
                lens_data = all_results[x]["latent_logit_lens"][p_idx]["logit_lens"]
                final_token = lens_data[-1]["top_indices"][0]
                layer_token = lens_data[li]["top_indices"][0]
                if layer_token == final_token:
                    match_count += 1
            stability[li, p_idx] = match_count / num_x

    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.cm.coolwarm
    colors = [cmap(i / max(1, num_positions - 1)) for i in range(num_positions)]

    for pos in range(num_positions):
        label = f"Latent {pos}"
        ax.plot(
            range(num_layers), stability[:, pos], "o-",
            color=colors[pos], markersize=4, linewidth=1.5,
            label=label,
        )

    ax.set_xlabel("Layer", fontsize=13, fontweight="bold")
    ax.set_ylabel("Fraction matching final-layer top-1", fontsize=13, fontweight="bold")
    ax.set_title(
        "Top-1 Token Stability Across Layers",
        fontsize=15, fontweight="bold",
    )
    ax.set_xticks(range(num_layers))
    ax.set_xticklabels([str(l) for l in layer_numbers], fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = results_dir / "token_stability_across_layers.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_within_series_cosine_similarity(
    all_vectors, x_values, results_dir, x_display_name="X",
):
    """
    For each key, compute a 7x7 cosine similarity matrix of that key's 7
    latent vectors against each other, then display all matrices as a single
    multi-panel grid PNG.

    This shows how similar the latent representations are *within* a single
    reasoning chain (e.g. is Latent 0 similar to Latent 6? do adjacent
    positions have higher similarity?).

    Args:
        all_vectors: dict mapping key -> list of 7 (1,1,hidden_dim) tensors
        x_values: list of keys
        x_display_name: Label for subplot titles (e.g. "X" or "Rollout")
    """
    num_items = len(x_values)
    num_positions = len(all_vectors[x_values[0]])

    cols = min(num_items, 5)
    rows = (num_items + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 4, rows * 3.8),
        squeeze=False,
    )

    for item_idx, key in enumerate(x_values):
        ax = axes[item_idx // cols][item_idx % cols]

        # Stack all 7 vectors for this key: (7, hidden_dim)
        vecs = []
        for pos in range(num_positions):
            v = all_vectors[key][pos].flatten().float()
            vecs.append(v)
        vecs = torch.stack(vecs)

        # Pairwise cosine similarity (7x7)
        norms = vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        vecs_normed = vecs / norms
        cos_mat = (vecs_normed @ vecs_normed.T).numpy()

        im = ax.imshow(cos_mat, cmap="RdYlGn", aspect="equal", vmin=-1.0, vmax=1.0)

        # Annotate cells with values
        for i in range(num_positions):
            for j in range(num_positions):
                val = cos_mat[i, j]
                text_color = "white" if val < 0.93 else "black"
                ax.text(
                    j, i, f"{val:.3f}",
                    ha="center", va="center", color=text_color, fontsize=6,
                )

        ax.set_title(f"{x_display_name} {key}", fontsize=10)
        ax.set_xticks(range(num_positions))
        ax.set_xticklabels([str(i) for i in range(num_positions)], fontsize=8)
        ax.set_yticks(range(num_positions))
        ax.set_yticklabels([str(i) for i in range(num_positions)], fontsize=8)
        ax.set_xlabel("Latent Vector Index", fontsize=8)
        ax.set_ylabel("Latent Vector Index", fontsize=8)

    # Hide unused axes
    for idx in range(num_items, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    # Add shared colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Cosine similarity", fontsize=10)

    plt.suptitle(
        f"Within-Series Cosine Similarity of Latent Vectors (per {x_display_name})",
        fontsize=14, fontweight="bold", y=1.01,
    )

    path = results_dir / "within_series_cosine_similarity.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")
