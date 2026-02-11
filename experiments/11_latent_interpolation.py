# ABOUTME: Latent vector interpolation experiment with logit lens analysis.
# ABOUTME: Interpolates between two prompts' latent reasoning vectors at a
# ABOUTME: specified position, then completes reasoning from each interpolated
# ABOUTME: starting point to see how answers transition.

# %%
import json
import sys
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import extract_answer_number
from src.interpolation import lerp, slerp
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

RESULTS_DIR = Path(__file__).parent.parent / "results" / "latent_interpolation"

# ============================================================================
# Prompt pair presets
# ============================================================================

PROMPT_PAIRS = {
    "number_change": {
        "prompt1": (
            "A shop has 12 items. They receive 2 more items. "
            "How many items does the shop have now? "
            "Give the answer only and nothing else."
        ),
        "prompt2": (
            "A shop has 12 items. They receive 8 more items. "
            "How many items does the shop have now? "
            "Give the answer only and nothing else."
        ),
        "ground_truth1": 14,
        "ground_truth2": 20,
    },
    "operation_switch": {
        "prompt1": (
            "A shop has 12 items. They receive 8 more items. "
            "How many items does the shop have now? "
            "Give the answer only and nothing else."
        ),
        "prompt2": (
            "A shop has 12 items. They sell 8 of the items. "
            "How many items does the shop have now? "
            "Give the answer only and nothing else."
        ),
        "ground_truth1": 20,
        "ground_truth2": 4,
    },
}


# ============================================================================
# Model Utilities (same as experiment 10)
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
# KV Cache Helpers
# ============================================================================


def clone_kv_cache(past_key_values):
    """Deep-clone a DynamicCache so the original is not mutated."""
    cloned = DynamicCache()
    for layer_idx in range(len(past_key_values.key_cache)):
        cloned.key_cache.append(past_key_values.key_cache[layer_idx].clone())
        cloned.value_cache.append(past_key_values.value_cache[layer_idx].clone())
    cloned._seen_tokens = past_key_values.get_seq_length()
    return cloned


# ============================================================================
# Collect latent vectors from a single prompt
# ============================================================================


def collect_latent_vectors(
    model, tokenizer, prompt, num_latent_iterations, save_kv_at_positions=None
):
    """
    Run a prompt through prefill + all K latent iterations and collect:
    - The post-projection latent vector at each step (list of K tensors)
    - Optionally, a snapshot of the KV cache BEFORE the specified latent positions

    The KV cache snapshot at position p means: the cache state after prefill +
    latent steps 0..p-1, so that injecting a vector at position p into that cache
    continues from the right point.

    Args:
        model: CODI model.
        tokenizer: Tokenizer.
        prompt: Input prompt string.
        num_latent_iterations: Number of latent steps (K).
        save_kv_at_positions: Set of latent position indices at which to snapshot
            the KV cache. None = don't save any.

    Returns:
        latent_vectors: List of K tensors, each (1, 1, hidden_dim).
        kv_snapshots: Dict mapping position -> DynamicCache clone.
            The cache at position p is the state BEFORE step p is executed.
    """
    device = model.codi.device
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)

    if save_kv_at_positions is None:
        save_kv_at_positions = set()

    latent_vectors = []
    kv_snapshots = {}

    with torch.no_grad():
        # Prefill
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=None,
            attention_mask=attention_mask,
        )
        past_kv = outputs.past_key_values

        # Initial latent embedding
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        # Project
        if model.use_prj:
            latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        # Save the initial latent vector (position 0, post-projection)
        latent_vectors.append(latent_embd.clone())

        # Save KV cache before position 0 if requested
        # (this is the cache after prefill, before any latent step)
        if 0 in save_kv_at_positions:
            kv_snapshots[0] = clone_kv_cache(past_kv)

        # Latent iterations
        for i in range(num_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_kv,
            )
            past_kv = outputs.past_key_values

            # Next latent embedding
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

            latent_vectors.append(latent_embd.clone())

            # Save KV cache before position (i+1) if requested
            # (cache now includes prefill + steps 0..i)
            if (i + 1) in save_kv_at_positions:
                kv_snapshots[i + 1] = clone_kv_cache(past_kv)

    return latent_vectors, kv_snapshots


# ============================================================================
# Run a single interpolation rollout
# ============================================================================


def run_interpolation_rollout(
    model,
    tokenizer,
    base_kv_cache,
    v_interp,
    remaining_iterations,
    lm_head,
    layer_norm,
    top_k_tokens=10,
    logit_lens_layers=None,
    max_new_tokens=128,
):
    """
    Given a KV cache (P1's cache up to position p) and an interpolated latent
    vector, continue the remaining latent steps and generate an answer.

    Args:
        base_kv_cache: KV cache to clone and use (NOT mutated).
        v_interp: Interpolated latent vector, shape (1, 1, hidden_dim).
        remaining_iterations: How many more latent steps after the interpolated one.
        lm_head: LM head for logit lens.
        layer_norm: Layer norm for logit lens.
        top_k_tokens: Top-K for logit lens.
        logit_lens_layers: Which layers to analyze. None = all.
        max_new_tokens: Max answer tokens.

    Returns:
        dict with generated_text, generated_tokens, latent_logit_lens.
    """
    device = model.codi.device
    embed_fn = model.get_embd(model.codi, model.model_name)
    vocab_size = model.codi.config.vocab_size
    eos_token_id = tokenizer.eos_token_id

    latent_logit_lens = []

    with torch.no_grad():
        past_kv = clone_kv_cache(base_kv_cache)
        latent_embd = v_interp

        # --- Feed the interpolated vector as the current latent step ---
        outputs = model.codi(
            inputs_embeds=latent_embd,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_kv,
        )
        past_kv = outputs.past_key_values

        # Logit lens on the interpolated step
        lens_result = logit_lens(
            outputs.hidden_states, lm_head, layer_norm,
            top_k=top_k_tokens, layer_indices=logit_lens_layers,
        )
        latent_logit_lens.append({"position": "interp", "logit_lens": lens_result})

        # Next latent embedding
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        if model.use_prj:
            latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        # --- Continue remaining latent iterations normally ---
        for i in range(remaining_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_kv,
            )
            past_kv = outputs.past_key_values

            lens_result = logit_lens(
                outputs.hidden_states, lm_head, layer_norm,
                top_k=top_k_tokens, layer_indices=logit_lens_layers,
            )
            latent_logit_lens.append(
                {"position": f"interp+{i + 1}", "logit_lens": lens_result}
            )

            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        # --- Generate answer tokens ---
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
        "generated_text": generated_text,
        "generated_tokens": generated_tokens,
        "latent_logit_lens": latent_logit_lens,
    }


# ============================================================================
# Visualization
# ============================================================================


def visualize_answer_vs_alpha(results, ground_truth1, ground_truth2, results_dir):
    """
    Plot numerical answer as a function of interpolation alpha.
    Horizontal lines show the two ground truth answers.
    """
    alphas = [r["alpha"] for r in results]
    answers = []
    for r in results:
        a = r.get("answer")
        if a is not None and a != float("inf"):
            answers.append(float(a))
        else:
            answers.append(float("nan"))

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot answers
    ax.plot(alphas, answers, "o-", color="dodgerblue", markersize=8, linewidth=2,
            label="Model answer", zorder=3)

    # Ground truth lines
    ax.axhline(y=ground_truth1, color="green", linestyle="--", linewidth=1.5,
               label=f"P1 ground truth = {ground_truth1}", alpha=0.7)
    ax.axhline(y=ground_truth2, color="red", linestyle="--", linewidth=1.5,
               label=f"P2 ground truth = {ground_truth2}", alpha=0.7)

    # Annotate each point with the answer
    for x, y, r in zip(alphas, answers, results):
        if not np.isnan(y):
            ax.annotate(
                r.get("generated_text", "")[:8],
                (x, y),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=8,
                color="gray",
            )

    ax.set_xlabel("Alpha (0 = P1, 1 = P2)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Numerical Answer", fontsize=13, fontweight="bold")
    ax.set_title("Answer vs Interpolation Alpha", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(alphas)

    plt.tight_layout()
    output_path = results_dir / "answer_vs_alpha.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def visualize_logit_lens_grid(results, tokenizer, results_dir):
    """
    One panel per alpha value. Each panel shows (analyzed_layer x latent_position)
    with top-1 token annotations.
    """
    num_alphas = len(results)
    num_positions = len(results[0]["latent_logit_lens"])

    # Get analyzed layer indices from first result
    analyzed_layers = [
        entry["layer"]
        for entry in results[0]["latent_logit_lens"][0]["logit_lens"]
    ]
    num_analyzed = len(analyzed_layers)

    cols = min(num_alphas, 4)
    rows = (num_alphas + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 5, rows * 4),
        squeeze=False,
    )

    for a_idx, r in enumerate(results):
        ax = axes[a_idx // cols][a_idx % cols]

        prob_matrix = np.zeros((num_analyzed, num_positions))
        token_matrix = [[None] * num_positions for _ in range(num_analyzed)]

        for p_idx, pos_data in enumerate(r["latent_logit_lens"]):
            for row_idx, layer_data in enumerate(pos_data["logit_lens"]):
                top_token_id = layer_data["top_indices"][0]
                top_prob = layer_data["top_probs"][0]
                prob_matrix[row_idx, p_idx] = top_prob
                token_matrix[row_idx][p_idx] = tokenizer.decode([top_token_id])

        im = ax.imshow(prob_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

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

        alpha_val = r["alpha"]
        answer_str = r.get("generated_text", "")[:12]
        ax.set_title(f"α={alpha_val:.2f}  ans={answer_str}", fontsize=10)
        ax.set_xlabel("Latent Pos", fontsize=9)
        ax.set_ylabel("Layer", fontsize=9)
        ax.set_xticks(range(num_positions))
        pos_labels = [str(d["position"]) for d in r["latent_logit_lens"]]
        ax.set_xticklabels(pos_labels, fontsize=7)
        ax.set_yticks(range(num_analyzed))
        ax.set_yticklabels([str(l) for l in analyzed_layers], fontsize=7)

    # Hide unused axes
    for idx in range(num_alphas, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    plt.tight_layout()
    output_path = results_dir / "logit_lens_grid.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def visualize_top1_comparison(results, tokenizer, results_dir):
    """
    Single heatmap: rows = alpha values, columns = latent positions.
    Each cell shows the top-1 token from the final analyzed layer.
    """
    num_alphas = len(results)
    num_positions = len(results[0]["latent_logit_lens"])

    token_matrix = []
    prob_matrix = np.zeros((num_alphas, num_positions))

    for a_idx, r in enumerate(results):
        row_tokens = []
        for p_idx, pos_data in enumerate(r["latent_logit_lens"]):
            final_layer = pos_data["logit_lens"][-1]
            top_token_id = final_layer["top_indices"][0]
            top_prob = final_layer["top_probs"][0]
            token_str = tokenizer.decode([top_token_id])
            row_tokens.append(token_str)
            prob_matrix[a_idx, p_idx] = top_prob
        token_matrix.append(row_tokens)

    fig, ax = plt.subplots(figsize=(3 + num_positions * 1.8, 2 + num_alphas * 0.7))

    im = ax.imshow(prob_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Top-1 probability", rotation=270, labelpad=15, fontsize=12)

    for i in range(num_alphas):
        for j in range(num_positions):
            token = token_matrix[i][j]
            prob = prob_matrix[i, j]
            token_display = repr(token)[1:-1] if token else ""
            text_color = "white" if prob > 0.5 else "black"
            ax.text(
                j, i, token_display,
                ha="center", va="center", color=text_color, fontsize=10,
            )

    alphas = [r["alpha"] for r in results]
    ax.set_xlabel("Latent Position", fontsize=13, fontweight="bold")
    ax.set_ylabel("Alpha", fontsize=13, fontweight="bold")
    ax.set_xticks(range(num_positions))
    pos_labels = [str(d["position"]) for d in results[0]["latent_logit_lens"]]
    ax.set_xticklabels(pos_labels, fontsize=10)
    ax.set_yticks(range(num_alphas))
    ax.set_yticklabels([f"{a:.2f}" for a in alphas], fontsize=10)
    ax.set_title(
        "Top-1 Logit Lens Token (Final Layer) vs Alpha",
        fontsize=14, fontweight="bold", pad=12,
    )

    plt.tight_layout()
    output_path = results_dir / "top1_comparison.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_summary_table(results, tokenizer):
    """Print a console summary table: alpha, answer, top-1 token at each latent position."""
    num_positions = len(results[0]["latent_logit_lens"])

    header = f"{'Alpha':>8s}"
    for p_idx, pos_data in enumerate(results[0]["latent_logit_lens"]):
        pos_label = str(pos_data["position"])
        header += f" | {pos_label:>12s}"
    header += f" | {'Answer':>12s}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for r in results:
        row = f"{r['alpha']:>8.2f}"
        for pos_data in r["latent_logit_lens"]:
            final_layer = pos_data["logit_lens"][-1]
            top_token_id = final_layer["top_indices"][0]
            top_prob = final_layer["top_probs"][0]
            token_str = repr(tokenizer.decode([top_token_id]))[1:-1]
            row += f" | {token_str:>8s} {top_prob:.2f}"
        answer_str = r.get("generated_text", "")[:12]
        row += f" | {answer_str:>12s}"
        print(row)

    print("=" * len(header))


# ============================================================================
# Main
# ============================================================================


def main(
    pair: str = "number_change",
    prompt1: str | None = None,
    prompt2: str | None = None,
    ground_truth1: int | None = None,
    ground_truth2: int | None = None,
    interp_position: int = 0,
    num_interp: int = 11,
    interp_method: str = "slerp",
    num_latent_iterations: int = NUM_LATENT_ITERATIONS,
    seed: int = 42,
    top_k: int = 10,
    max_new_tokens: int = 128,
    logit_lens_layers: list[int] | None = None,
):
    """
    Interpolate between two prompts' latent vectors at a specified position
    and observe how the model's answer transitions.

    Args:
        pair: Preset prompt pair name ("number_change", "operation_switch")
            or "custom" to use prompt1/prompt2/ground_truth1/ground_truth2.
        prompt1: First prompt (only used when pair="custom").
        prompt2: Second prompt (only used when pair="custom").
        ground_truth1: Expected answer for prompt1 (only when pair="custom").
        ground_truth2: Expected answer for prompt2 (only when pair="custom").
        interp_position: Which latent step to interpolate at (0 = first latent).
        num_interp: Number of interpolation points (e.g. 11 for alpha=0.0..1.0).
        interp_method: "slerp" or "lerp".
        num_latent_iterations: Total number of latent reasoning steps (K).
        seed: Random seed (for reproducibility of model loading).
        top_k: Top-K tokens for logit lens.
        max_new_tokens: Max answer tokens to generate.
        logit_lens_layers: Layer indices for logit lens. None = all.
    """
    load_dotenv()

    # Resolve prompt pair
    if pair == "custom":
        assert prompt1 is not None and prompt2 is not None, (
            "Must provide prompt1 and prompt2 when pair='custom'"
        )
        assert ground_truth1 is not None and ground_truth2 is not None, (
            "Must provide ground_truth1 and ground_truth2 when pair='custom'"
        )
    else:
        preset = PROMPT_PAIRS[pair]
        prompt1 = preset["prompt1"]
        prompt2 = preset["prompt2"]
        ground_truth1 = preset["ground_truth1"]
        ground_truth2 = preset["ground_truth2"]

    # Resolve interpolation function
    interp_fn = slerp if interp_method == "slerp" else lerp

    results_dir = RESULTS_DIR / f"{pair}_pos{interp_position}_{interp_method}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Resolve logit lens layer indices
    num_model_layers = None  # will be set after model loads

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

    # Print config
    print(f"\nPair: {pair}")
    print(f"P1: {prompt1}")
    print(f"P2: {prompt2}")
    print(f"Ground truth: P1={ground_truth1}, P2={ground_truth2}")
    print(f"Interpolation position: {interp_position}")
    print(f"Num interpolation points: {num_interp}")
    print(f"Method: {interp_method}")
    print(f"Num latent iterations: {num_latent_iterations}")
    print(f"Logit lens layers: {logit_lens_layers or 'all'}")

    # ---- Step 1: Collect latent vectors from both prompts ----
    print(f"\nCollecting latent vectors from P1...")
    p1_vectors, p1_kv_snapshots = collect_latent_vectors(
        model, tokenizer, prompt1, num_latent_iterations,
        save_kv_at_positions={interp_position},
    )
    print(f"  Got {len(p1_vectors)} latent vectors")

    print(f"Collecting latent vectors from P2...")
    p2_vectors, _ = collect_latent_vectors(
        model, tokenizer, prompt2, num_latent_iterations,
        save_kv_at_positions=None,  # don't need P2's cache
    )
    print(f"  Got {len(p2_vectors)} latent vectors")

    # Get the two vectors to interpolate between
    v1 = p1_vectors[interp_position]
    v2 = p2_vectors[interp_position]

    # Cosine similarity between the two vectors
    cos_sim = torch.nn.functional.cosine_similarity(
        v1.flatten(), v2.flatten(), dim=0
    ).item()
    print(f"\nCosine similarity between v1[{interp_position}] and v2[{interp_position}]: {cos_sim:.4f}")

    # Get the KV cache to use (P1's cache at the interpolation position)
    base_kv = p1_kv_snapshots[interp_position]

    # How many latent steps remain after the interpolated one
    # Total steps: positions 0..K-1 (K vectors fed, producing K forward passes)
    # If interp_position=0: we feed the interpolated vector, then K-1 more
    # If interp_position=p: we feed the interpolated vector, then K-1-p more
    remaining_iters = num_latent_iterations - 1 - interp_position
    # But the first latent vector (position 0) gets projected and fed,
    # then num_latent_iterations forward passes happen in the loop.
    # So total latent positions = 1 (initial) + num_latent_iterations (loop).
    # Actually let me recount based on experiment 10's structure:
    # - Position 0: initial latent (from prefill output, projected)
    # - Positions 1..K: from the loop (K = num_latent_iterations)
    # Total: K+1 latent vectors
    # If interp at position p (0-indexed), remaining = K - p
    remaining_iters = num_latent_iterations - interp_position
    print(f"Remaining latent iterations after interpolation: {remaining_iters}")

    # ---- Step 2: Run interpolation rollouts ----
    alphas = np.linspace(0.0, 1.0, num_interp).tolist()
    all_results = []

    for alpha in alphas:
        print(f"  Running alpha={alpha:.3f}...")
        v_interp = interp_fn(v1, v2, alpha)

        rollout = run_interpolation_rollout(
            model=model,
            tokenizer=tokenizer,
            base_kv_cache=base_kv,
            v_interp=v_interp,
            remaining_iterations=remaining_iters,
            lm_head=lm_head,
            layer_norm=layer_norm,
            top_k_tokens=top_k,
            logit_lens_layers=resolved_layers,
            max_new_tokens=max_new_tokens,
        )

        answer = extract_answer_number(rollout["generated_text"])
        rollout["alpha"] = alpha
        rollout["answer"] = answer
        rollout["correct_p1"] = (
            answer is not None
            and answer != float("inf")
            and int(answer) == ground_truth1
        )
        rollout["correct_p2"] = (
            answer is not None
            and answer != float("inf")
            and int(answer) == ground_truth2
        )

        print(
            f"    answer={answer}  text={rollout['generated_text']!r}  "
            f"(P1={rollout['correct_p1']}, P2={rollout['correct_p2']})"
        )

        all_results.append(rollout)

    # ---- Summary ----
    print("\n")
    print_summary_table(all_results, tokenizer)

    num_p1 = sum(1 for r in all_results if r["correct_p1"])
    num_p2 = sum(1 for r in all_results if r["correct_p2"])
    print(f"\nCorrect as P1: {num_p1}/{num_interp}")
    print(f"Correct as P2: {num_p2}/{num_interp}")

    # ---- Save JSON ----
    json_results = {
        "config": {
            "pair": pair,
            "prompt1": prompt1,
            "prompt2": prompt2,
            "ground_truth1": ground_truth1,
            "ground_truth2": ground_truth2,
            "interp_position": interp_position,
            "num_interp": num_interp,
            "interp_method": interp_method,
            "num_latent_iterations": num_latent_iterations,
            "seed": seed,
            "cosine_similarity": cos_sim,
            "logit_lens_layers": logit_lens_layers,
        },
        "results": [],
    }

    for r in all_results:
        json_results["results"].append({
            "alpha": r["alpha"],
            "generated_text": r["generated_text"],
            "answer": r["answer"],
            "correct_p1": r["correct_p1"],
            "correct_p2": r["correct_p2"],
            "latent_logit_lens": r["latent_logit_lens"],
        })

    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    # ---- Visualizations ----
    print("\nCreating visualizations...")
    visualize_answer_vs_alpha(all_results, ground_truth1, ground_truth2, results_dir)
    visualize_top1_comparison(all_results, tokenizer, results_dir)
    visualize_logit_lens_grid(all_results, tokenizer, results_dir)

    print("\nExperiment complete!")


# %%
if __name__ == "__main__":
    fire.Fire(main)
