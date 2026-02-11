# ABOUTME: Input-space interpolation experiment. Varies a single number X in a
# ABOUTME: math word problem (X=2..20), runs each through CODI, collects latent
# ABOUTME: vectors + answers, then analyzes with logit lens, t-SNE, cosine
# ABOUTME: similarity to find monotonic patterns in the latent reasoning space.

# %%
import json
import sys
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.manifold import TSNE
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import extract_answer_number
from src.model import CODI

# %%
# Parameters
CHECKPOINT_PATH = "bcywinski/codi_llama1b-answer_only"
MODEL_NAME_OR_PATH = "meta-llama/Llama-3.2-1B-Instruct"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

DTYPE = "bfloat16"

NUM_LATENT_ITERATIONS = 6

RESULTS_DIR = Path(__file__).parent.parent / "results" / "input_space_interpolation"

# Default prompt template: {X} is replaced with the varying number
DEFAULT_TEMPLATE = (
    "A shop has 12 items. They receive {X} more items. "
    "How many items does the shop have now? "
    "Give the answer only and nothing else."
)


# ============================================================================
# Model Utilities (shared with experiments 10/11)
# ============================================================================


def get_lm_head(model):
    return model.codi.get_base_model().lm_head


def get_layer_norm(model):
    return model.codi.get_base_model().model.norm


def logit_lens(hidden_states, lm_head, layer_norm, top_k=5, layer_indices=None):
    """
    Apply logit lens to hidden states for a single (batch=1) forward pass.
    Returns a list of dicts, one per analyzed layer.
    """
    if layer_indices is None:
        indices_to_use = range(len(hidden_states))
    else:
        indices_to_use = layer_indices

    results = []
    for layer_idx in indices_to_use:
        h = hidden_states[layer_idx]
        h_last = h[:, -1, :]  # (1, hidden)
        h_last = layer_norm(h_last)
        logits = lm_head(h_last)  # (1, vocab)
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

        results.append(
            {
                "layer": layer_idx,
                "top_indices": top_indices[0].cpu().tolist(),
                "top_probs": top_probs[0].cpu().tolist(),
            }
        )

    return results


def prepare_inputs(model, tokenizer, prompt):
    """Construct input sequence: [Prompt Tokens] + [EOS] + [BOCOT]"""
    device = model.codi.device

    inputs = tokenizer(
        prompt, return_tensors="pt", padding=False, add_special_tokens=True
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    sot_id = tokenizer.convert_tokens_to_ids("<|bocot|>")
    eos_id = tokenizer.eos_token_id

    bot_tensor = torch.tensor([[eos_id, sot_id]], dtype=torch.long, device=device)
    input_ids_bot = torch.cat([input_ids, bot_tensor], dim=1)
    attention_mask_bot = torch.cat(
        [attention_mask, torch.ones_like(bot_tensor)], dim=1
    )

    return input_ids_bot, attention_mask_bot


def ensure_tokenizer_special_tokens(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|bocot|>", "<|eocot|>"]}
    )


# ============================================================================
# Collect latent vectors + logit lens + answer from a single prompt
# ============================================================================


def run_full_pass(
    model,
    tokenizer,
    prompt,
    num_latent_iterations,
    lm_head,
    layer_norm,
    top_k_tokens=10,
    logit_lens_layers=None,
    max_new_tokens=128,
):
    """
    Run a single prompt through prefill + K latent iterations + answer generation.

    Returns:
        dict with:
          - latent_vectors: list of K+1 tensors (1, 1, hidden_dim), detached on CPU
          - latent_logit_lens: list of K+1 logit lens results per latent position
          - generated_text: decoded answer string
          - generated_tokens: list of token ids
    """
    device = model.codi.device
    embed_fn = model.get_embd(model.codi, model.model_name)
    vocab_size = model.codi.config.vocab_size
    eos_token_id = tokenizer.eos_token_id

    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)

    latent_vectors = []
    latent_logit_lens = []

    with torch.no_grad():
        # ---- Prefill ----
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=None,
            attention_mask=attention_mask,
        )
        past_kv = outputs.past_key_values

        # Initial latent embedding (position 0) — extracted from prefill
        # NOTE: no logit lens here; prefill hidden states reflect <|bocot|>
        # token prediction (always <|eocot|>), not the thought vector content.
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        if model.use_prj:
            latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        latent_vectors.append(latent_embd.detach().cpu().clone())

        # ---- Latent iterations ----
        # Each iteration feeds vector i and produces vector i+1.
        # The logit lens at iteration i captures the model's internal state
        # while processing vector i — labelled "position i".
        for i in range(num_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_kv,
            )
            past_kv = outputs.past_key_values

            # Logit lens for vector i (the vector we just fed in)
            lens_result = logit_lens(
                outputs.hidden_states, lm_head, layer_norm,
                top_k=top_k_tokens, layer_indices=logit_lens_layers,
            )
            latent_logit_lens.append(
                {"position": i, "logit_lens": lens_result}
            )

            # Extract vector i+1
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

            latent_vectors.append(latent_embd.detach().cpu().clone())

        # ---- Logit lens for the final vector (position K) ----
        # Clone KV cache so this extra forward pass doesn't affect answer gen.
        cloned_kv = DynamicCache()
        for layer_idx in range(len(past_kv)):
            key, value = past_kv[layer_idx]
            cloned_kv.update(key.clone(), value.clone(), layer_idx)

        outputs_final = model.codi(
            inputs_embeds=latent_embd,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=cloned_kv,
        )
        lens_result = logit_lens(
            outputs_final.hidden_states, lm_head, layer_norm,
            top_k=top_k_tokens, layer_indices=logit_lens_layers,
        )
        latent_logit_lens.append(
            {"position": num_latent_iterations, "logit_lens": lens_result}
        )

        # ---- Generate answer tokens ----
        eot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")
        eot_ids = torch.tensor([[eot_id]], dtype=torch.long, device=device)
        output = embed_fn(eot_ids)

        generated_tokens = []

        for _ in range(max_new_tokens):
            out = model.codi(
                inputs_embeds=output,
                output_hidden_states=False,
                attention_mask=None,
                use_cache=True,
                past_key_values=past_kv,
            )
            past_kv = out.past_key_values
            logits = out.logits[:, -1, : vocab_size - 1]

            next_token_id = torch.argmax(logits, dim=-1).item()
            generated_tokens.append(next_token_id)

            if next_token_id == eos_token_id:
                break

            output = embed_fn(
                torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            )

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    return {
        "latent_vectors": latent_vectors,
        "latent_logit_lens": latent_logit_lens,
        "generated_text": generated_text,
        "generated_tokens": generated_tokens,
    }


# ============================================================================
# Analysis Utilities
# ============================================================================


def compute_cosine_similarity_matrices(all_vectors, x_values):
    """
    For each latent position, compute a pairwise cosine similarity matrix
    across all X values.

    Args:
        all_vectors: dict mapping X -> list of (1,1,hidden_dim) tensors
        x_values: list of X values

    Returns:
        dict mapping position_index -> (N x N) numpy array of cosine similarities
    """
    num_positions = len(all_vectors[x_values[0]])
    N = len(x_values)
    cos_matrices = {}

    for pos in range(num_positions):
        # Collect vectors at this position across all X values
        vecs = []
        for x in x_values:
            v = all_vectors[x][pos].flatten().float()
            vecs.append(v)
        vecs = torch.stack(vecs)  # (N, hidden_dim)

        # Pairwise cosine similarity
        norms = vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        vecs_normed = vecs / norms
        cos_mat = (vecs_normed @ vecs_normed.T).numpy()
        cos_matrices[pos] = cos_mat

    return cos_matrices


def compute_consecutive_cosine_sims(all_vectors, x_values):
    """
    For each latent position, compute cosine similarity between consecutive
    X values: sim(X=k, X=k+1). Tests monotonicity/smoothness.

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
    X value's vector and every other X value's vector.

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
# Visualizations
# ============================================================================


def visualize_answer_vs_x(all_results, x_values, results_dir):
    """Plot model's numerical answer vs input X, with ground truth line y = 12 + X."""
    answers = []
    for x in x_values:
        a = all_results[x].get("answer")
        if a is not None and a != float("inf"):
            answers.append(float(a))
        else:
            answers.append(float("nan"))

    ground_truths = [12 + x for x in x_values]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(x_values, answers, "o-", color="dodgerblue", markersize=8, linewidth=2,
            label="Model answer", zorder=3)
    ax.plot(x_values, ground_truths, "s--", color="green", markersize=6,
            linewidth=1.5, label="Ground truth (12 + X)", alpha=0.7)

    # Annotate each point
    for x, y, r_x in zip(x_values, answers, x_values):
        if not np.isnan(y):
            text = all_results[r_x].get("generated_text", "")[:8]
            ax.annotate(
                text, (x, y),
                textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=7, color="gray",
            )

    ax.set_xlabel("X (number added)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Numerical Answer", fontsize=13, fontweight="bold")
    ax.set_title("Model Answer vs Input Number X", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_values)

    plt.tight_layout()
    path = results_dir / "answer_vs_x.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_logit_lens_heatmap(all_results, x_values, tokenizer, results_dir):
    """
    Heatmap: rows = X values, columns = latent positions.
    Each cell shows top-1 token from the final analyzed layer.
    """
    num_x = len(x_values)
    num_positions = len(all_results[x_values[0]]["latent_logit_lens"])

    token_matrix = []
    prob_matrix = np.zeros((num_x, num_positions))

    for row_idx, x in enumerate(x_values):
        row_tokens = []
        for p_idx, pos_data in enumerate(all_results[x]["latent_logit_lens"]):
            final_layer = pos_data["logit_lens"][-1]
            top_token_id = final_layer["top_indices"][0]
            top_prob = final_layer["top_probs"][0]
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

    ax.set_xlabel("Latent Position", fontsize=13, fontweight="bold")
    ax.set_ylabel("X (input number)", fontsize=13, fontweight="bold")
    ax.set_xticks(range(num_positions))
    pos_labels = [
        str(d["position"])
        for d in all_results[x_values[0]]["latent_logit_lens"]
    ]
    ax.set_xticklabels(pos_labels, fontsize=10)
    ax.set_yticks(range(num_x))
    ax.set_yticklabels([str(x) for x in x_values], fontsize=10)
    ax.set_title(
        "Top-1 Logit Lens Token (Final Layer) vs Input X",
        fontsize=14, fontweight="bold", pad=12,
    )

    plt.tight_layout()
    path = results_dir / "logit_lens_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_cosine_similarity_matrices(cos_matrices, x_values, results_dir):
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

        ax.set_title(f"Latent Position {pos}", fontsize=11)
        ax.set_xlabel("X", fontsize=9)
        ax.set_ylabel("X", fontsize=9)

        tick_positions = list(range(0, len(x_values), max(1, len(x_values) // 8)))
        tick_labels = [str(x_values[i]) for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=7)

    for idx in range(num_positions, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    plt.suptitle(
        "Pairwise Cosine Similarity of Latent Vectors Across Input X",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    path = results_dir / "cosine_similarity_matrices.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_drift_from_first(drift, x_values, results_dir):
    """
    Line plot: for each latent position, how does cosine similarity to X_min
    change as X increases? Tests monotonic drift.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    cmap = plt.cm.viridis
    num_positions = len(drift)
    colors = [cmap(i / max(1, num_positions - 1)) for i in range(num_positions)]

    for pos in range(num_positions):
        ax.plot(
            x_values, drift[pos], "o-",
            color=colors[pos], markersize=5, linewidth=1.5,
            label=f"Position {pos}",
        )

    ax.set_xlabel("X (input number)", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"Cosine similarity to X={x_values[0]}", fontsize=13, fontweight="bold")
    ax.set_title(
        "Latent Vector Drift from First Input",
        fontsize=15, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_values)

    plt.tight_layout()
    path = results_dir / "drift_from_first.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_consecutive_cosine_sims(consec_sims, x_values, results_dir):
    """
    Line plot: cosine similarity between consecutive X values for each
    latent position. Flat = smooth latent space, dips = phase transitions.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    cmap = plt.cm.viridis
    num_positions = len(consec_sims)
    colors = [cmap(i / max(1, num_positions - 1)) for i in range(num_positions)]

    x_midpoints = [(x_values[i] + x_values[i + 1]) / 2 for i in range(len(x_values) - 1)]

    for pos in range(num_positions):
        ax.plot(
            x_midpoints, consec_sims[pos], "o-",
            color=colors[pos], markersize=5, linewidth=1.5,
            label=f"Position {pos}",
        )

    ax.set_xlabel("X (midpoint between consecutive inputs)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cosine similarity (X, X+1)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Consecutive Cosine Similarity Between Adjacent Inputs",
        fontsize=15, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = results_dir / "consecutive_cosine_similarity.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_tsne(all_vectors, x_values, results_dir, perplexity=5, seed=42):
    """
    t-SNE plot of all latent vectors. Points colored by latent position,
    with lines connecting same-position points across X values.
    Also creates a second plot colored by X value.
    """
    num_positions = len(all_vectors[x_values[0]])
    N = len(x_values)

    # Flatten all vectors into a matrix: (N * num_positions, hidden_dim)
    all_vecs = []
    labels_position = []
    labels_x = []
    for x in x_values:
        for pos in range(num_positions):
            v = all_vectors[x][pos].flatten().float().numpy()
            all_vecs.append(v)
            labels_position.append(pos)
            labels_x.append(x)

    all_vecs = np.array(all_vecs)
    labels_position = np.array(labels_position)
    labels_x = np.array(labels_x)

    # Adjust perplexity if we have very few samples
    effective_perplexity = min(perplexity, max(2, len(all_vecs) // 4))

    tsne = TSNE(
        n_components=2, perplexity=effective_perplexity,
        random_state=seed, max_iter=2000, learning_rate="auto", init="pca",
    )
    embeddings = tsne.fit_transform(all_vecs)

    # ---- Plot 1: colored by latent position, lines connect same position ----
    fig, ax = plt.subplots(figsize=(12, 9))

    cmap = plt.cm.tab10
    for pos in range(num_positions):
        mask = labels_position == pos
        idxs = np.where(mask)[0]
        color = cmap(pos % 10)

        # Draw connecting lines (in order of X)
        ax.plot(
            embeddings[idxs, 0], embeddings[idxs, 1],
            "-", color=color, linewidth=1.0, alpha=0.4, zorder=1,
        )

        # Draw points
        ax.scatter(
            embeddings[idxs, 0], embeddings[idxs, 1],
            c=[color], s=50, label=f"Pos {pos}", zorder=2, edgecolors="white",
            linewidth=0.5,
        )

        # Annotate first and last X
        ax.annotate(
            f"X={x_values[0]}", (embeddings[idxs[0], 0], embeddings[idxs[0], 1]),
            fontsize=6, color=color, alpha=0.8,
        )
        ax.annotate(
            f"X={x_values[-1]}", (embeddings[idxs[-1], 0], embeddings[idxs[-1], 1]),
            fontsize=6, color=color, alpha=0.8,
        )

    ax.set_xlabel("t-SNE dim 1", fontsize=12)
    ax.set_ylabel("t-SNE dim 2", fontsize=12)
    ax.set_title(
        "t-SNE of Latent Vectors (colored by position, connected across X)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = results_dir / "tsne_by_position.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")

    # ---- Plot 2: colored by X value, lines connect same X across positions ----
    fig, ax = plt.subplots(figsize=(12, 9))

    x_min, x_max = min(x_values), max(x_values)
    norm = plt.Normalize(vmin=x_min, vmax=x_max)
    cmap_x = plt.cm.coolwarm

    for x_idx, x in enumerate(x_values):
        mask = labels_x == x
        idxs = np.where(mask)[0]
        color = cmap_x(norm(x))

        # Draw connecting lines (positions 0, 1, ..., K in order)
        ax.plot(
            embeddings[idxs, 0], embeddings[idxs, 1],
            "-", color=color, linewidth=1.0, alpha=0.4, zorder=1,
        )

        ax.scatter(
            embeddings[idxs, 0], embeddings[idxs, 1],
            c=[color], s=50, zorder=2, edgecolors="white", linewidth=0.5,
        )

    # Add colorbar for X values
    sm = plt.cm.ScalarMappable(cmap=cmap_x, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("X (input number)", fontsize=12)

    ax.set_xlabel("t-SNE dim 1", fontsize=12)
    ax.set_ylabel("t-SNE dim 2", fontsize=12)
    ax.set_title(
        "t-SNE of Latent Vectors (colored by input X, connected across positions)",
        fontsize=14, fontweight="bold",
    )
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = results_dir / "tsne_by_x_value.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_vector_norms(all_vectors, x_values, results_dir):
    """
    Plot L2 norms of latent vectors across X values for each position.
    Checks whether vector magnitude changes systematically with X.
    """
    num_positions = len(all_vectors[x_values[0]])

    fig, ax = plt.subplots(figsize=(12, 5))

    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, num_positions - 1)) for i in range(num_positions)]

    for pos in range(num_positions):
        norms = []
        for x in x_values:
            v = all_vectors[x][pos].flatten().float()
            norms.append(v.norm().item())
        ax.plot(
            x_values, norms, "o-",
            color=colors[pos], markersize=5, linewidth=1.5,
            label=f"Position {pos}",
        )

    ax.set_xlabel("X (input number)", fontsize=13, fontweight="bold")
    ax.set_ylabel("L2 Norm", fontsize=13, fontweight="bold")
    ax.set_title(
        "Latent Vector L2 Norms vs Input X",
        fontsize=15, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_values)

    plt.tight_layout()
    path = results_dir / "vector_norms.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def print_summary_table(all_results, x_values, tokenizer, base_number):
    """Print a console summary: X, answer, top-1 token at each latent position."""
    num_positions = len(all_results[x_values[0]]["latent_logit_lens"])

    header = f"{'X':>5s} | {'GT':>5s}"
    for p_idx in range(num_positions):
        pos_label = all_results[x_values[0]]["latent_logit_lens"][p_idx]["position"]
        header += f" | {'Lat ' + str(pos_label):>12s}"
    header += f" | {'Answer':>12s}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for x in x_values:
        r = all_results[x]
        gt = base_number + x
        answer_str = r.get("generated_text", "")[:12]
        row = f"{x:>5d} | {gt:>5d}"
        for pos_data in r["latent_logit_lens"]:
            final_layer = pos_data["logit_lens"][-1]
            top_token_id = final_layer["top_indices"][0]
            top_prob = final_layer["top_probs"][0]
            token_str = repr(tokenizer.decode([top_token_id]))[1:-1]
            row += f" | {token_str:>8s} {top_prob:.2f}"
        row += f" | {answer_str:>12s}"
        print(row)

    print("=" * len(header))


# ============================================================================
# Main
# ============================================================================


def main(
    template: str = DEFAULT_TEMPLATE,
    x_start: int = 2,
    x_end: int = 20,
    base_number: int = 12,
    num_latent_iterations: int = NUM_LATENT_ITERATIONS,
    seed: int = 42,
    top_k: int = 10,
    max_new_tokens: int = 128,
    logit_lens_layers: list[int] | None = None,
    tsne_perplexity: float = 5.0,
):
    """
    Input-space interpolation: vary a single number X in a math word problem,
    run each variant through CODI, and analyze how latent vectors change.

    Args:
        template: Prompt template with {X} placeholder.
        x_start: Starting value for X (inclusive).
        x_end: Ending value for X (inclusive).
        base_number: The base number in the problem (for ground truth = base + X).
        num_latent_iterations: Number of latent reasoning steps (K).
        seed: Random seed.
        top_k: Top-K tokens for logit lens.
        max_new_tokens: Max answer tokens.
        logit_lens_layers: Layer indices for logit lens (None = all).
        tsne_perplexity: Perplexity for t-SNE.
    """
    load_dotenv()

    x_values = list(range(x_start, x_end + 1))
    results_dir = RESULTS_DIR / f"x{x_start}_to_{x_end}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Model setup
    print("Loading model...")
    torch.manual_seed(seed)
    model = CODI.from_pretrained(
        checkpoint_path=CHECKPOINT_PATH,
        model_name_or_path=MODEL_NAME_OR_PATH,
        lora_r=128,
        lora_alpha=32,
        num_latent=6,
        use_prj=True,
        device=DEVICE,
        dtype=DTYPE,
        strict=False,
        checkpoint_save_path=f"./checkpoints/{CHECKPOINT_PATH}",
        remove_eos=False,
        full_precision=True,
    )
    tokenizer = model.tokenizer
    ensure_tokenizer_special_tokens(tokenizer)

    lm_head = get_lm_head(model)
    layer_norm = get_layer_norm(model)

    num_model_layers = model.codi.config.num_hidden_layers + 1
    if logit_lens_layers is not None:
        resolved_layers = [
            l if l >= 0 else num_model_layers + l for l in logit_lens_layers
        ]
    else:
        resolved_layers = None

    # Config
    print(f"\nTemplate: {template}")
    print(f"X range: {x_start} to {x_end} ({len(x_values)} prompts)")
    print(f"Base number: {base_number}")
    print(f"Num latent iterations: {num_latent_iterations}")
    print(f"Logit lens layers: {logit_lens_layers or 'all'}")

    # ---- Run all prompts ----
    all_results = {}  # X -> result dict
    all_vectors = {}  # X -> list of latent vectors (CPU tensors)

    for x in x_values:
        prompt = template.replace("{X}", str(x))
        ground_truth = base_number + x
        print(f"\n  X={x:>2d}  prompt: {prompt[:60]}...")

        result = run_full_pass(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            num_latent_iterations=num_latent_iterations,
            lm_head=lm_head,
            layer_norm=layer_norm,
            top_k_tokens=top_k,
            logit_lens_layers=resolved_layers,
            max_new_tokens=max_new_tokens,
        )

        answer = extract_answer_number(result["generated_text"])
        result["x"] = x
        result["prompt"] = prompt
        result["ground_truth"] = ground_truth
        result["answer"] = answer
        result["correct"] = (
            answer is not None
            and answer != float("inf")
            and int(answer) == ground_truth
        )

        print(
            f"         answer={answer}  text={result['generated_text']!r}  "
            f"correct={result['correct']}  (GT={ground_truth})"
        )

        all_vectors[x] = result.pop("latent_vectors")  # keep vectors separate
        all_results[x] = result

    # ---- Summary ----
    print("\n")
    print_summary_table(all_results, x_values, tokenizer, base_number)

    num_correct = sum(1 for x in x_values if all_results[x]["correct"])
    print(f"\nCorrect: {num_correct}/{len(x_values)}")

    # ---- Analysis ----
    print("\nComputing cosine similarity matrices...")
    cos_matrices = compute_cosine_similarity_matrices(all_vectors, x_values)

    print("Computing consecutive cosine similarities...")
    consec_sims = compute_consecutive_cosine_sims(all_vectors, x_values)

    print("Computing drift from first input...")
    drift = compute_drift_from_first(all_vectors, x_values)

    # ---- Save JSON ----
    json_results = {
        "config": {
            "template": template,
            "x_start": x_start,
            "x_end": x_end,
            "base_number": base_number,
            "num_latent_iterations": num_latent_iterations,
            "seed": seed,
            "logit_lens_layers": logit_lens_layers,
        },
        "results": [],
    }

    for x in x_values:
        r = all_results[x]
        json_results["results"].append({
            "x": x,
            "prompt": r["prompt"],
            "ground_truth": r["ground_truth"],
            "generated_text": r["generated_text"],
            "answer": r["answer"],
            "correct": r["correct"],
            "latent_logit_lens": r["latent_logit_lens"],
        })

    # Add analysis summaries
    json_results["analysis"] = {
        "consecutive_cosine_sims": {
            str(pos): sims for pos, sims in consec_sims.items()
        },
        "drift_from_first": {
            str(pos): sims for pos, sims in drift.items()
        },
    }

    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    # ---- Visualizations ----
    print("\nCreating visualizations...")
    visualize_answer_vs_x(all_results, x_values, results_dir)
    visualize_logit_lens_heatmap(all_results, x_values, tokenizer, results_dir)
    visualize_cosine_similarity_matrices(cos_matrices, x_values, results_dir)
    visualize_drift_from_first(drift, x_values, results_dir)
    visualize_consecutive_cosine_sims(consec_sims, x_values, results_dir)
    visualize_vector_norms(all_vectors, x_values, results_dir)

    print("Running t-SNE...")
    visualize_tsne(all_vectors, x_values, results_dir,
                   perplexity=tsne_perplexity, seed=seed)

    print("\nExperiment complete!")


# %%
if __name__ == "__main__":
    fire.Fire(main)
