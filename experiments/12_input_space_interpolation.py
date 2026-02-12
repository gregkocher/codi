# ABOUTME: Input-space interpolation experiment. Varies a single number X in a
# ABOUTME: math word problem (X=2..20), runs each through CODI, collects latent
# ABOUTME: vectors + answers, then analyzes with logit lens, t-SNE, cosine
# ABOUTME: similarity to find monotonic patterns in the latent reasoning space.

# %%
import importlib
import json
import sys
from pathlib import Path

import fire
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from dotenv import load_dotenv
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.datasets import extract_answer_number
from src.model import CODI
from src.templates import ADDITION_FIRST_TEMPLATES, SUBTRACTION_FIRST_TEMPLATES

# Import shared utilities from experiment 3
_exp3 = importlib.import_module("3_logit_lens_latents")
get_lm_head = _exp3.get_lm_head
get_layer_norm = _exp3.get_layer_norm
logit_lens = _exp3.logit_lens
prepare_inputs = _exp3.prepare_inputs
ensure_tokenizer_special_tokens = _exp3.ensure_tokenizer_special_tokens

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


# ============================================================================
# Preset prompt configurations
# ============================================================================

# The addition/subtraction templates use {X}, {Y}, {Z} as variable names.
# In our presets we pick one to sweep (replaced with {X} in the final template)
# and fix the others.

PRESETS = {
    "simple_add": {
        "description": "Simple addition: base + X",
        "template": (
            "A shop has 12 items. They receive {X} more items. "
            "How many items does the shop have now? "
            "Give the answer only and nothing else."
        ),
        "gt_formula": "12 + X",
        "default_fixed_vars": {"base": 12},
        "default_sweep_var": "X",
        "default_x_start": 2,
        "default_x_end": 20,
        "default_extra_x": [25, 30, 50, 100, 500, 1000, 1999, 2026, 5000, 100000],
    },
    "addition": {
        "description": "Addition template: (A+B)*(C+1), sweep one of A/B/C",
        "raw_template": ADDITION_FIRST_TEMPLATES[0],  # uses {X}, {Y}, {Z}
        "gt_formula": "(A+B)*(C+1)",
        "default_fixed_vars": {"A": 3, "C": 2},
        "default_sweep_var": "B",
        "default_x_start": 1,
        "default_x_end": 8,
        "default_extra_x": None,
    },
    "subtraction": {
        "description": "Subtraction template: (A-B)*(C+1), sweep A (starting members)",
        "raw_template": SUBTRACTION_FIRST_TEMPLATES[0],  # uses {X}, {Y}, {Z}
        "gt_formula": "(A-B)*(C+1)",
        "default_fixed_vars": {"B": 2, "C": 2},
        "default_sweep_var": "A",
        "default_x_start": 3,
        "default_x_end": 20,
        "default_extra_x": [25, 30, 50, 100, 500, 1000, 5000, 10000],
    },
}

# Map from our sweep variable names (A, B, C) to the template placeholders ({X}, {Y}, {Z})
_SWEEP_TO_TEMPLATE_VAR = {"A": "X", "B": "Y", "C": "Z"}


def build_template_from_preset(preset_name: str, sweep_var: str, fixed_vars: dict) -> str:
    """
    Build a prompt template string with {X} as the only remaining placeholder.

    For 'simple_add', returns the template as-is.
    For 'addition'/'subtraction', takes the raw 3-variable template, substitutes
    the fixed variables with their values, and renames the swept variable to {X}.
    """
    preset = PRESETS[preset_name]

    if preset_name == "simple_add":
        return preset["template"]

    raw = preset["raw_template"]
    # The raw template uses {X}, {Y}, {Z} (the repo's convention).
    # Our variables are A, B, C which map to X, Y, Z respectively.
    # First, substitute the fixed variables.
    for var_name, value in fixed_vars.items():
        tpl_var = _SWEEP_TO_TEMPLATE_VAR[var_name]
        raw = raw.replace("{" + tpl_var + "}", str(value))

    # Now rename the swept variable's placeholder to {X} (our sweep placeholder)
    swept_tpl_var = _SWEEP_TO_TEMPLATE_VAR[sweep_var]
    raw = raw.replace("{" + swept_tpl_var + "}", "{X}")

    return raw


def compute_ground_truth(x: int, preset_name: str, sweep_var: str, fixed_vars: dict) -> int:
    """
    Compute the ground truth answer for a given X value and preset.

    Args:
        x: The swept variable's value.
        preset_name: One of 'simple_add', 'addition', 'subtraction'.
        sweep_var: Which variable is being swept ('A', 'B', or 'C').
        fixed_vars: Dict of fixed variable values (e.g. {'A': 3, 'C': 2}).

    Returns:
        Integer ground truth answer.
    """
    if preset_name == "simple_add":
        return fixed_vars.get("base", 12) + x

    # Resolve A, B, C values
    var_values = dict(fixed_vars)
    var_values[sweep_var] = x
    a, b, c = var_values["A"], var_values["B"], var_values["C"]

    if preset_name == "addition":
        step_1 = a + b
    elif preset_name == "subtraction":
        step_1 = a - b
    else:
        raise ValueError(f"Unknown preset: {preset_name}")

    return step_1 * c + step_1  # = step_1 * (c + 1)


def gt_label(preset_name: str, sweep_var: str, fixed_vars: dict) -> str:
    """Return a human-readable label for the ground truth formula."""
    if preset_name == "simple_add":
        base = fixed_vars.get("base", 12)
        return f"{base} + X"

    var_strs = {}
    for v in ("A", "B", "C"):
        if v == sweep_var:
            var_strs[v] = "X"
        else:
            var_strs[v] = str(fixed_vars[v])

    if preset_name == "addition":
        return f"({var_strs['A']}+{var_strs['B']})*({var_strs['C']}+1)"
    elif preset_name == "subtraction":
        return f"({var_strs['A']}-{var_strs['B']})*({var_strs['C']}+1)"
    return "?"


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

    Indexing:
      - latent_logit_lens[0] = "Latent 0" (prefill hidden states, last token = <|bocot|>)
      - latent_logit_lens[1..K] = "Latent 1" .. "Latent K" (loop iterations)
      - latent_vectors[0] = initial embedding from prefill
      - latent_vectors[1..K] = outputs from each latent iteration

    Returns:
        dict with:
          - latent_vectors: list of K+1 tensors (1, 1, hidden_dim), detached on CPU
          - latent_logit_lens: list of K+1 logit lens results (Latent 0 .. Latent K)
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

        # Initial latent embedding — extracted from prefill
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        if model.use_prj:
            latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        latent_vectors.append(latent_embd.detach().cpu().clone())

        # Logit lens on prefill output (latent vector 0)
        lens_result = logit_lens(
            outputs.hidden_states, lm_head, layer_norm,
            top_k=top_k_tokens, layer_indices=logit_lens_layers,
        )
        latent_logit_lens.append({"position": "Latent 0", "logit_lens": lens_result})

        # ---- Latent iterations ----
        for i in range(num_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_kv,
            )
            past_kv = outputs.past_key_values

            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

            latent_vectors.append(latent_embd.detach().cpu().clone())

            lens_result = logit_lens(
                outputs.hidden_states, lm_head, layer_norm,
                top_k=top_k_tokens, layer_indices=logit_lens_layers,
            )
            latent_logit_lens.append(
                {"position": f"Latent {i + 1}", "logit_lens": lens_result}
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


def visualize_answer_vs_x(all_results, x_values, gt_label_str, results_dir):
    """Plot model's numerical answer vs input X, with ground truth line."""
    # Use evenly-spaced indices so large X values don't distort the axis
    indices = list(range(len(x_values)))
    x_labels = [str(x) for x in x_values]

    answers = []
    for x in x_values:
        a = all_results[x].get("answer")
        if a is not None and a != float("inf"):
            answers.append(float(a))
        else:
            answers.append(float("nan"))

    ground_truths = [all_results[x]["ground_truth"] for x in x_values]

    fig, ax = plt.subplots(figsize=(max(12, len(x_values) * 0.5), 5))

    ax.plot(indices, answers, "o-", color="dodgerblue", markersize=8, linewidth=2,
            label="Model answer", zorder=3)
    ax.plot(indices, ground_truths, "s--", color="green", markersize=6,
            linewidth=1.5, label=f"Ground truth ({gt_label_str})", alpha=0.7)

    # Annotate each point
    for idx, (x, y) in enumerate(zip(x_values, answers)):
        if not np.isnan(y):
            text = all_results[x].get("generated_text", "")[:12]
            ax.annotate(
                text, (idx, y),
                textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=6, color="gray", rotation=45,
            )

    ax.set_xlabel("X (number added)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Numerical Answer", fontsize=13, fontweight="bold")
    ax.set_title("Model Answer vs Input Number X", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(indices)
    ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")

    plt.tight_layout()
    path = results_dir / "answer_vs_x.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_logit_lens_heatmap(
    all_results, x_values, tokenizer, results_dir, layer_index=None,
):
    """
    Heatmap: rows = X values, columns = latent vector index (0..K).
    Each cell shows top-1 token from a specific layer's logit lens.

    Args:
        layer_index: Index into the logit_lens list for each position.
            None (default) = last layer (-1).
    """
    num_x = len(x_values)
    num_positions = len(all_results[x_values[0]]["latent_logit_lens"])

    # Determine which layer entry to use
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
    ax.set_ylabel("X (input number)", fontsize=13, fontweight="bold")
    ax.set_xticks(range(num_positions))
    ax.set_xticklabels([str(i) for i in range(num_positions)], fontsize=10)
    ax.set_yticks(range(num_x))
    ax.set_yticklabels([str(x) for x in x_values], fontsize=10)

    if layer_index is None:
        title = "Top-1 Logit Lens Token (Final Layer) vs Input X"
        filename = "logit_lens_heatmap.png"
    else:
        title = f"Top-1 Logit Lens Token (Layer {actual_layer}) vs Input X"
        filename = f"logit_lens_heatmap_layer_{actual_layer}.png"

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    path = results_dir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_logit_lens_heatmap_top5(
    all_results, x_values, tokenizer, results_dir, layer_index=None,
):
    """
    Heatmap: rows = X values, columns = latent vector index (0..K).
    Each cell shows the top-1 token prominently and tokens 2-5 in smaller
    font below in parentheses.

    Args:
        layer_index: Index into the logit_lens list for each position.
            None (default) = last layer (-1).
    """
    num_x = len(x_values)
    num_positions = len(all_results[x_values[0]]["latent_logit_lens"])

    li = -1 if layer_index is None else layer_index

    # Collect top-5 tokens and top-1 probability per cell
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

    # Taller rows to accommodate the extra text
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
    ax.set_ylabel("X (input number)", fontsize=13, fontweight="bold")
    ax.set_xticks(range(num_positions))
    ax.set_xticklabels([str(i) for i in range(num_positions)], fontsize=10)
    ax.set_yticks(range(num_x))
    ax.set_yticklabels([str(x) for x in x_values], fontsize=10)

    if layer_index is None:
        title = "Top-5 Logit Lens Tokens (Final Layer) vs Input X"
        filename = "logit_lens_heatmap_top5.png"
    else:
        title = f"Top-5 Logit Lens Tokens (Layer {actual_layer}) vs Input X"
        filename = f"logit_lens_heatmap_top5_layer_{actual_layer}.png"

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    path = results_dir / filename
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

        label = f"Latent {pos}"
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("X", fontsize=9)
        ax.set_ylabel("X", fontsize=9)

        all_ticks = list(range(len(x_values)))
        all_labels = [str(x) for x in x_values]
        ax.set_xticks(all_ticks)
        ax.set_xticklabels(all_labels, fontsize=6, rotation=45, ha="right")
        ax.set_yticks(all_ticks)
        ax.set_yticklabels(all_labels, fontsize=6)

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

    ax.set_xlabel("X (input number)", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"Cosine similarity to X={x_values[0]}", fontsize=13, fontweight="bold")
    ax.set_title(
        "Latent Vector Drift from First Input",
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


def visualize_consecutive_cosine_sims(consec_sims, x_values, results_dir):
    """
    Line plot: cosine similarity between consecutive X values for each
    latent position. Flat = smooth latent space, dips = phase transitions.
    """
    # Labels for midpoints between consecutive pairs
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

    ax.set_xlabel("Consecutive X pair", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cosine similarity (X_i, X_{i+1})", fontsize=13, fontweight="bold")
    ax.set_title(
        "Consecutive Cosine Similarity Between Adjacent Inputs",
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
    """Return a color normalizer for X values; uses LogNorm when range > 100x."""
    x_min, x_max = min(x_values), max(x_values)
    if x_min > 0 and x_max / x_min > 100:
        return mcolors.LogNorm(vmin=x_min, vmax=x_max)
    return plt.Normalize(vmin=x_min, vmax=x_max)


def _plot_embeddings_by_position(
    embeddings, labels_position, x_values, num_positions, ax, method_name,
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
            f"X={x_values[0]}", (embeddings[idxs[0], 0], embeddings[idxs[0], 1]),
            fontsize=6, color=color, alpha=0.8,
        )
        ax.annotate(
            f"X={x_values[-1]}", (embeddings[idxs[-1], 0], embeddings[idxs[-1], 1]),
            fontsize=6, color=color, alpha=0.8,
        )

    ax.set_xlabel(f"{method_name} dim 1", fontsize=12)
    ax.set_ylabel(f"{method_name} dim 2", fontsize=12)
    ax.set_title(
        f"{method_name} of Latent Vectors (colored by position, connected across X)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.2)


def _plot_embeddings_by_x(
    embeddings, labels_x, x_values, ax, method_name,
):
    """Plot 2-D embeddings colored by X value with connecting lines."""
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
    cbar.set_label("X (input number)", fontsize=12)

    ax.set_xlabel(f"{method_name} dim 1", fontsize=12)
    ax.set_ylabel(f"{method_name} dim 2", fontsize=12)
    ax.set_title(
        f"{method_name} of Latent Vectors (colored by input X, connected across positions)",
        fontsize=14, fontweight="bold",
    )
    ax.grid(True, alpha=0.2)


def visualize_tsne(all_vectors, x_values, results_dir, perplexity=5, seed=42):
    """
    t-SNE plot of all latent vectors. Points colored by latent position,
    with lines connecting same-position points across X values.
    Also creates a second plot colored by X value.
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
    )
    plt.tight_layout()
    path = results_dir / "tsne_by_position.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")

    # Plot 2: colored by X value
    fig, ax = plt.subplots(figsize=(12, 9))
    _plot_embeddings_by_x(embeddings, labels_x, x_values, ax, "t-SNE")
    plt.tight_layout()
    path = results_dir / "tsne_by_x_value.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_umap(all_vectors, x_values, results_dir, n_neighbors=15, min_dist=0.1, seed=42):
    """
    UMAP plot of all latent vectors. Same two sub-plots as t-SNE:
    colored by latent position and colored by X value.
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
    )
    plt.tight_layout()
    path = results_dir / "umap_by_position.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")

    # Plot 2: colored by X value
    fig, ax = plt.subplots(figsize=(12, 9))
    _plot_embeddings_by_x(embeddings, labels_x, x_values, ax, "UMAP")
    plt.tight_layout()
    path = results_dir / "umap_by_x_value.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_vector_norms(all_vectors, x_values, results_dir):
    """
    Plot L2 norms of latent vectors across X values for each position.
    Checks whether vector magnitude changes systematically with X.
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

    ax.set_xlabel("X (input number)", fontsize=13, fontweight="bold")
    ax.set_ylabel("L2 Norm", fontsize=13, fontweight="bold")
    ax.set_title(
        "Latent Vector L2 Norms vs Input X",
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


def visualize_token_first_occurrence_depth(all_results, x_values, tokenizer, results_dir):
    """
    Heatmap showing the earliest layer at which the top-1 token matches the
    final layer's top-1 token and never changes again (first occurrence depth).
    Rows = X values, columns = latent vector index.
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
    ax.set_ylabel("X (input number)", fontsize=13, fontweight="bold")
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


def visualize_layer_entropy(all_results, x_values, results_dir):
    """
    Heatmap: rows = layers, columns = latent vector index.
    Each cell = mean entropy (over top-K probs) across all X values.
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
        "Layer-wise Mean Entropy of Logit Lens (across X values)",
        fontsize=13, fontweight="bold", pad=12,
    )

    plt.tight_layout()
    path = results_dir / "layer_entropy_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_token_stability_across_layers(all_results, x_values, results_dir):
    """
    Line chart: for each latent position, what fraction of X values have the
    same top-1 token as the final layer, at each layer depth?
    """
    num_positions = len(all_results[x_values[0]]["latent_logit_lens"])
    sample_lens = all_results[x_values[0]]["latent_logit_lens"][0]["logit_lens"]
    num_layers = len(sample_lens)
    layer_numbers = [ld["layer"] for ld in sample_lens]
    num_x = len(x_values)

    # stability_matrix[li][pos] = fraction of X values matching final top-1
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


def print_summary_table(all_results, x_values, tokenizer):
    """Print a console summary: X, answer, top-1 token at each latent position."""
    num_positions = len(all_results[x_values[0]]["latent_logit_lens"])

    header = f"{'X':>5s} | {'GT':>5s}"
    for p_idx in range(num_positions):
        pos_label = str(all_results[x_values[0]]["latent_logit_lens"][p_idx]["position"])
        header += f" | {pos_label:>12s}"
    header += f" | {'Answer':>12s}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for x in x_values:
        r = all_results[x]
        gt = r["ground_truth"]
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
    preset: str = "simple_add",
    sweep_var: str | None = None,
    fixed_vars: dict | None = None,
    template: str | None = None,
    x_start: int | None = None,
    x_end: int | None = None,
    extra_x: list[int] | None = "default",
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
        preset: Prompt preset ('simple_add', 'addition', 'subtraction').
        sweep_var: Which variable to sweep. Defaults per preset
            (simple_add: 'X', addition/subtraction: 'B').
        fixed_vars: Fixed variable values (e.g. {'A': 3, 'C': 2}).
            Defaults per preset.
        template: Override prompt template with {X} placeholder (ignores preset).
        x_start: Starting value for X (inclusive). Defaults per preset.
        x_end: Ending value for X (inclusive). Defaults per preset.
        extra_x: Additional X values to append after the range. Use 'default'
            for preset default, None for no extras.
        num_latent_iterations: Number of latent reasoning steps (K).
        seed: Random seed.
        top_k: Top-K tokens for logit lens.
        max_new_tokens: Max answer tokens.
        logit_lens_layers: Layer indices for logit lens (None = all).
        tsne_perplexity: Perplexity for t-SNE.
    """
    load_dotenv()

    # ---- Resolve preset defaults ----
    assert preset in PRESETS, f"Unknown preset: {preset}. Choose from {list(PRESETS.keys())}"
    p = PRESETS[preset]

    if sweep_var is None:
        sweep_var = p["default_sweep_var"]
    if fixed_vars is None:
        fixed_vars = dict(p["default_fixed_vars"])
    else:
        fixed_vars = dict(fixed_vars)  # copy
    if x_start is None:
        x_start = p["default_x_start"]
    if x_end is None:
        x_end = p["default_x_end"]
    if extra_x == "default":
        extra_x = p["default_extra_x"]

    # Build the template (user override takes priority)
    if template is None:
        template = build_template_from_preset(preset, sweep_var, fixed_vars)

    # Build the ground truth label
    gt_label_str = gt_label(preset, sweep_var, fixed_vars)

    x_values = list(range(x_start, x_end + 1))
    if extra_x:
        existing = set(x_values)
        for v in extra_x:
            if v not in existing:
                x_values.append(v)
                existing.add(v)

    x_max_label = x_values[-1] if extra_x else x_end
    results_dir = RESULTS_DIR / f"{preset}_x{x_start}_to_{x_max_label}"
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
    print(f"\nPreset: {preset} — {p['description']}")
    print(f"Template: {template}")
    print(f"Sweep var: {sweep_var}, Fixed vars: {fixed_vars}")
    print(f"GT formula: {gt_label_str}")
    print(f"X values: {x_values[0]}..{x_values[-1]} ({len(x_values)} prompts)")
    if extra_x:
        print(f"  (includes extra: {[v for v in x_values if v > x_end]})")
    print(f"Num latent iterations: {num_latent_iterations}")
    print(f"Logit lens layers: {logit_lens_layers or 'all'}")

    # ---- Run all prompts ----
    all_results = {}  # X -> result dict
    all_vectors = {}  # X -> list of latent vectors (CPU tensors)

    for x in x_values:
        prompt = template.replace("{X}", str(x))
        ground_truth = compute_ground_truth(x, preset, sweep_var, fixed_vars)
        print(f"\n  X={x}  prompt: {prompt[:60]}...")

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
    print_summary_table(all_results, x_values, tokenizer)

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
            "preset": preset,
            "sweep_var": sweep_var,
            "fixed_vars": fixed_vars,
            "gt_formula": gt_label_str,
            "template": template,
            "x_start": x_start,
            "x_end": x_end,
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
    visualize_answer_vs_x(all_results, x_values, gt_label_str, results_dir)
    visualize_logit_lens_heatmap(all_results, x_values, tokenizer, results_dir)
    visualize_logit_lens_heatmap_top5(all_results, x_values, tokenizer, results_dir)
    visualize_cosine_similarity_matrices(cos_matrices, x_values, results_dir)
    visualize_drift_from_first(drift, x_values, results_dir)
    visualize_consecutive_cosine_sims(consec_sims, x_values, results_dir)
    visualize_vector_norms(all_vectors, x_values, results_dir)

    # ---- Per-layer logit lens heatmaps ----
    num_ll_layers = len(
        all_results[x_values[0]]["latent_logit_lens"][0]["logit_lens"]
    )
    print(f"\nGenerating per-layer logit lens heatmaps ({num_ll_layers} layers)...")
    for li in range(num_ll_layers):
        visualize_logit_lens_heatmap(
            all_results, x_values, tokenizer, results_dir, layer_index=li,
        )
        visualize_logit_lens_heatmap_top5(
            all_results, x_values, tokenizer, results_dir, layer_index=li,
        )

    # ---- Cross-layer analysis ----
    print("\nCross-layer analysis...")
    visualize_token_first_occurrence_depth(
        all_results, x_values, tokenizer, results_dir,
    )
    visualize_layer_entropy(all_results, x_values, results_dir)
    visualize_token_stability_across_layers(
        all_results, x_values, results_dir,
    )

    print("Running t-SNE...")
    visualize_tsne(all_vectors, x_values, results_dir,
                   perplexity=tsne_perplexity, seed=seed)

    print("Running UMAP...")
    visualize_umap(all_vectors, x_values, results_dir, seed=seed)

    print("\nExperiment complete!")


# %%
if __name__ == "__main__":
    fire.Fire(main)
