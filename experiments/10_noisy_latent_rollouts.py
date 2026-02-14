# ABOUTME: Noisy latent rollout experiment with logit lens analysis (batched).
# ABOUTME: Runs N rollouts on the same prompt in a single batched forward pass,
# ABOUTME: with norm-preserving noise at specified latent positions.
# ABOUTME: Collects latent vectors and uses shared analysis/viz from exp_utils.

# %%
import importlib
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
sys.path.insert(0, str(Path(__file__).parent))

from src.datasets import extract_answer_number
from src.model import CODI
from src.noise import add_norm_preserving_noise

# Import shared utilities from experiment 3
_exp3 = importlib.import_module("3_logit_lens_latents")
get_lm_head = _exp3.get_lm_head
get_layer_norm = _exp3.get_layer_norm
logit_lens = _exp3.logit_lens
prepare_inputs = _exp3.prepare_inputs
ensure_tokenizer_special_tokens = _exp3.ensure_tokenizer_special_tokens

# Import shared analysis / visualization utilities
from exp_utils import (
    compute_consecutive_cosine_sims,
    compute_cosine_similarity_matrices,
    compute_drift_from_first,
    visualize_consecutive_cosine_sims,
    visualize_cosine_similarity_matrices,
    visualize_drift_from_first,
    visualize_full_logit_lens_grid,
    visualize_layer_entropy,
    visualize_logit_lens_heatmap,
    visualize_logit_lens_heatmap_top5,
    visualize_token_first_occurrence_depth,
    visualize_token_stability_across_layers,
    visualize_tsne,
    visualize_umap,
    visualize_vector_norms,
    visualize_within_series_cosine_similarity,
)

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

DEFAULT_PROMPT = (
    "A team starts with 3 members. They recruit 5 new members. "
    "Then each current member recruits 2 additional people. "
    "How many people are there now on the team? "
    "Give the answer only and nothing else."
)
DEFAULT_GROUND_TRUTH = (3 + 5) * 2 + (3 + 5)  # = 24

RESULTS_DIR = Path(__file__).parent.parent / "results" / "noisy_latent_rollouts"


# ============================================================================
# KV Cache + Noise Helpers
# ============================================================================


def expand_kv_cache(past_key_values, batch_size):
    """
    Expand a DynamicCache from batch_size=1 to batch_size=N by repeating
    along the batch dimension.
    """
    expanded = DynamicCache()
    for layer_idx in range(len(past_key_values.key_cache)):
        # Each: (1, num_heads, seq_len, head_dim) -> (N, ...)
        key = past_key_values.key_cache[layer_idx].repeat(batch_size, 1, 1, 1)
        value = past_key_values.value_cache[layer_idx].repeat(batch_size, 1, 1, 1)
        expanded.key_cache.append(key)
        expanded.value_cache.append(value)
    expanded._seen_tokens = past_key_values.get_seq_length()
    return expanded


def add_per_element_noise(tensor, noise_scale, generators):
    """
    Add different norm-preserving noise to each batch element using
    per-element generators for reproducibility.

    Args:
        tensor: (batch, 1, hidden_dim) latent embeddings.
        noise_scale: Gaussian noise scale.
        generators: List of torch.Generator, one per batch element.

    Returns:
        Noisy tensor with same shape, each element noised independently.
    """
    result = tensor.clone()
    for b in range(tensor.shape[0]):
        result[b : b + 1] = add_norm_preserving_noise(
            tensor[b : b + 1], noise_scale, generator=generators[b]
        )
    return result


# ============================================================================
# Batched Rollouts with Noise + Logit Lens + Latent Vectors
# ============================================================================


def run_batched_noisy_rollouts(
    model,
    tokenizer,
    prompt,
    lm_head,
    layer_norm,
    noise_positions,
    noise_scale,
    base_seed,
    num_rollouts=10,
    num_latent_iterations=6,
    max_new_tokens=128,
    greedy=True,
    top_k_tokens=10,
    logit_lens_layers=None,
):
    """
    Run N rollouts in a single batched forward pass.

    Prefills the prompt once (batch=1), expands the KV cache to N, then runs
    latent iterations and answer generation as batch_size=N. Each rollout gets
    different noise via its own torch.Generator seeded with base_seed + rollout_idx.

    Indexing (matches experiment 12):
      - latent_logit_lens[0] = "Latent 0" (prefill hidden states — shared)
      - latent_logit_lens[1..K] = "Latent 1" .. "Latent K" (per-rollout)
      - latent_vectors[0] = initial embedding after projection + noise
      - latent_vectors[1..K] = outputs from each latent iteration after projection + noise

    Returns:
        List of N dicts, each with:
          - generated_text: decoded answer string
          - generated_tokens: list of token ids
          - latent_logit_lens: list of K+1 dicts (Latent 0..K)
          - latent_vectors: list of K+1 tensors (1, 1, hidden_dim) on CPU
    """
    device = model.codi.device
    N = num_rollouts
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)

    # Warn if no noise — all rollouts will be identical with greedy decoding
    if noise_positions is None and greedy:
        print(
            "WARNING: noise_positions=None with greedy=True → "
            "all rollouts will produce identical results."
        )

    # Create per-rollout generators for reproducible noise
    generators = []
    for k in range(N):
        gen = torch.Generator(device=device)
        gen.manual_seed(base_seed + k)
        generators.append(gen)

    # Resolve layer indices for logit lens (handle negative indices)
    num_layers = model.codi.config.num_hidden_layers + 1
    if logit_lens_layers is not None:
        resolved_layers = [
            l if l >= 0 else num_layers + l for l in logit_lens_layers
        ]
    else:
        resolved_layers = None

    # Cache frequently-accessed values
    embed_fn = model.get_embd(model.codi, model.model_name)
    vocab_size = model.codi.config.vocab_size
    eos_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or 0

    all_latent_logit_lens = [[] for _ in range(N)]
    all_latent_vectors = [[] for _ in range(N)]

    with torch.no_grad():
        # ----- Step 1: Prefill once (batch_size=1) -----
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=None,
            attention_mask=attention_mask,
        )

        # Logit lens on prefill output — "Latent 0" (shared across all rollouts)
        prefill_lens_result = logit_lens(
            outputs.hidden_states, lm_head, layer_norm,
            top_k=top_k_tokens, layer_indices=resolved_layers,
        )
        for b in range(N):
            all_latent_logit_lens[b].append(
                {"position": "Latent 0", "logit_lens": prefill_lens_result}
            )

        # ----- Step 2: Expand KV cache to batch_size=N -----
        past_kv = expand_kv_cache(outputs.past_key_values, N)

        # Initial latent embedding: (1, 1, hidden) -> (N, 1, hidden)
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        latent_embd = latent_embd.expand(N, -1, -1).clone()

        # Project
        if model.use_prj:
            latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        # Noise at position 0
        if noise_positions is not None and 0 in noise_positions:
            latent_embd = add_per_element_noise(
                latent_embd, noise_scale, generators
            )

        # Store Latent 0 vector for each rollout
        for b in range(N):
            all_latent_vectors[b].append(
                latent_embd[b : b + 1].detach().cpu().clone()
            )

        # ----- Step 3: Latent iterations (batched) -----
        for i in range(num_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_kv,
            )
            past_kv = outputs.past_key_values

            # Logit lens — per batch element using shared logit_lens
            for b in range(N):
                hs_b = tuple(h[b : b + 1] for h in outputs.hidden_states)
                lens_result = logit_lens(
                    hs_b, lm_head, layer_norm,
                    top_k=top_k_tokens, layer_indices=resolved_layers,
                )
                all_latent_logit_lens[b].append(
                    {"position": f"Latent {i + 1}", "logit_lens": lens_result}
                )

            # Next latent embedding
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

            # Noise at this position (1-indexed)
            if noise_positions is not None and (i + 1) in noise_positions:
                latent_embd = add_per_element_noise(
                    latent_embd, noise_scale, generators
                )

            # Store latent vector for each rollout
            for b in range(N):
                all_latent_vectors[b].append(
                    latent_embd[b : b + 1].detach().cpu().clone()
                )

        # ----- Step 4: Generate answer tokens (batched) -----
        eot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")
        eot_ids = torch.tensor([[eot_id]], dtype=torch.long, device=device)
        output = embed_fn(eot_ids).expand(N, -1, -1)

        finished = torch.zeros(N, dtype=torch.bool, device=device)
        generated_tokens = [[] for _ in range(N)]

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

            if greedy:
                next_token_ids = torch.argmax(logits, dim=-1)  # (N,)
            else:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

            for b in range(N):
                if not finished[b]:
                    generated_tokens[b].append(next_token_ids[b].item())
                    if next_token_ids[b] == eos_token_id:
                        finished[b] = True

            if finished.all():
                break

            # For finished sequences, force pad token to avoid garbage embeddings
            next_ids_for_emb = next_token_ids.clone()
            if finished.any():
                next_ids_for_emb[finished] = pad_id

            output = embed_fn(next_ids_for_emb.unsqueeze(1))

    # ----- Build per-rollout results -----
    results = []
    for b in range(N):
        text = tokenizer.decode(generated_tokens[b], skip_special_tokens=False)
        results.append(
            {
                "generated_text": text,
                "generated_tokens": generated_tokens[b],
                "latent_logit_lens": all_latent_logit_lens[b],
                "latent_vectors": all_latent_vectors[b],
            }
        )

    return results


# ============================================================================
# Experiment-10-specific helpers
# ============================================================================


def print_summary_table(all_rollout_results, tokenizer):
    """Print a console summary table: top-1 token at the final analyzed layer for each rollout x latent position."""
    num_positions = len(all_rollout_results[0]["latent_logit_lens"])

    header = f"{'Rollout':>8s}"
    for p in range(num_positions):
        header += f" | {'Latent ' + str(p):>12s}"
    header += f" | {'Answer':>12s} | {'Correct':>7s}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for r_idx, rollout in enumerate(all_rollout_results):
        row = f"{r_idx:>8d}"
        for p_idx, pos_data in enumerate(rollout["latent_logit_lens"]):
            final_layer = pos_data["logit_lens"][-1]
            top_token_id = final_layer["top_indices"][0]
            top_prob = final_layer["top_probs"][0]
            token_str = repr(tokenizer.decode([top_token_id]))[1:-1]
            row += f" | {token_str:>8s} {top_prob:.2f}"
        answer_str = rollout.get("generated_text", "")[:12]
        correct_str = "Y" if rollout.get("correct", False) else "N"
        row += f" | {answer_str:>12s} | {correct_str:>7s}"
        print(row)

    print("=" * len(header))


def visualize_accuracy_fraction(all_rollout_results, results_dir):
    """Bar chart showing the fraction of correct rollouts."""
    num_rollouts = len(all_rollout_results)
    num_correct = sum(1 for r in all_rollout_results if r.get("correct", False))
    frac = num_correct / num_rollouts if num_rollouts > 0 else 0

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["Correct", "Incorrect"],
        [num_correct, num_rollouts - num_correct],
        color=["#4CAF50", "#F44336"],
        edgecolor="black",
        linewidth=0.5,
    )

    # Annotate bars
    for bar, count in zip(bars, [num_correct, num_rollouts - num_correct]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(count), ha="center", va="bottom", fontsize=14, fontweight="bold",
        )

    ax.set_ylabel("Number of Rollouts", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Accuracy: {num_correct}/{num_rollouts} ({frac:.0%})",
        fontsize=15, fontweight="bold",
    )
    ax.set_ylim(0, num_rollouts + 1)

    plt.tight_layout()
    path = results_dir / "accuracy_fraction.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================================
# Main
# ============================================================================


def main(
    prompt: str = DEFAULT_PROMPT,
    ground_truth: int = DEFAULT_GROUND_TRUTH,
    noise_positions: list[int] | None = None,
    noise_scale: float = 0.1,
    seed: int = 42,
    num_rollouts: int = 10,
    top_k: int = 10,
    num_latent_iterations: int = NUM_LATENT_ITERATIONS,
    greedy: bool = True,
    max_new_tokens: int = 128,
    logit_lens_layers: list[int] | None = None,
    tsne_perplexity: float = 5.0,
):
    """
    Run N noisy rollouts on a single prompt with logit lens analysis (batched).

    All rollouts share a single prompt prefill, then diverge at the latent reasoning
    stage where each rollout receives different noise. This is ~N times faster than
    running rollouts sequentially.

    Args:
        prompt: The input prompt to evaluate.
        ground_truth: Expected integer answer for correctness checking.
        noise_positions: List of latent position indices to perturb (e.g. [0,2,3,5]).
            None means no noise (baseline).
        noise_scale: Standard deviation multiplier for Gaussian noise before renormalization.
        seed: Base random seed. Rollout k uses seed + k.
        num_rollouts: Number of rollouts to run.
        top_k: Number of top tokens for logit lens.
        num_latent_iterations: Number of latent reasoning iterations.
        greedy: Use greedy decoding for answer generation.
        max_new_tokens: Maximum tokens to generate for the answer.
        logit_lens_layers: Layer indices for logit lens. None = all layers.
            E.g. [-4,-3,-2,-1] for last 4 layers only (big speedup).
        tsne_perplexity: Perplexity for t-SNE.
    """
    load_dotenv()

    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    # Model setup
    print("Loading model...")
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

    # Print config
    print(f"\nPrompt: {prompt}")
    print(f"Ground truth: {ground_truth}")
    print(f"Noise positions: {noise_positions}")
    print(f"Noise scale: {noise_scale}")
    print(f"Seed: {seed}")
    print(f"Num rollouts: {num_rollouts}")
    print(f"Top-K tokens: {top_k}")
    print(f"Num latent iterations: {num_latent_iterations}")
    print(f"Greedy: {greedy}")
    print(f"Logit lens layers: {logit_lens_layers or 'all'}")
    print(f"Mode: BATCHED (all {num_rollouts} rollouts in one pass)")

    # Run all rollouts in a single batched pass
    print(f"\nRunning {num_rollouts} batched rollouts...")
    all_rollout_results = run_batched_noisy_rollouts(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        lm_head=lm_head,
        layer_norm=layer_norm,
        noise_positions=noise_positions,
        noise_scale=noise_scale,
        base_seed=seed,
        num_rollouts=num_rollouts,
        num_latent_iterations=num_latent_iterations,
        max_new_tokens=max_new_tokens,
        greedy=greedy,
        top_k_tokens=top_k,
        logit_lens_layers=logit_lens_layers,
    )

    # Post-process: check correctness and annotate results
    for rollout_idx, result in enumerate(all_rollout_results):
        answer = extract_answer_number(result["generated_text"])
        correct = (
            answer is not None
            and answer != float("inf")
            and int(answer) == ground_truth
        )
        result["answer"] = answer
        result["correct"] = correct
        result["rollout_idx"] = rollout_idx
        result["seed"] = seed + rollout_idx

        print(
            f"  Rollout {rollout_idx} (seed={result['seed']}): "
            f"{result['generated_text']!r}  answer={answer}  correct={correct}"
        )

    # Summary table
    print("\n")
    print_summary_table(all_rollout_results, tokenizer)

    # Accuracy summary
    num_correct = sum(1 for r in all_rollout_results if r["correct"])
    print(f"\nAccuracy: {num_correct}/{num_rollouts} ({num_correct / num_rollouts:.1%})")

    # ---- Reshape data for shared utils ----
    # Build dicts keyed by rollout index, matching experiment 12's data format
    rollout_keys = list(range(num_rollouts))  # [0, 1, 2, ..., N-1]

    all_results = {}   # rollout_idx -> result dict (with latent_logit_lens)
    all_vectors = {}   # rollout_idx -> list of K+1 tensors

    for r_idx, result in enumerate(all_rollout_results):
        all_vectors[r_idx] = result.pop("latent_vectors")
        all_results[r_idx] = result

    # ---- Save JSON ----
    json_results = {
        "config": {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "noise_positions": noise_positions,
            "noise_scale": noise_scale,
            "seed": seed,
            "num_rollouts": num_rollouts,
            "top_k": top_k,
            "num_latent_iterations": num_latent_iterations,
            "greedy": greedy,
            "max_new_tokens": max_new_tokens,
            "logit_lens_layers": logit_lens_layers,
            "mode": "batched",
        },
        "summary": {
            "num_correct": num_correct,
            "accuracy": num_correct / num_rollouts,
        },
        "rollouts": [],
    }

    for r_idx in rollout_keys:
        r = all_results[r_idx]
        rollout_data = {
            "rollout_idx": r["rollout_idx"],
            "seed": r["seed"],
            "generated_text": r["generated_text"],
            "answer": r["answer"],
            "correct": r["correct"],
            "latent_logit_lens": r["latent_logit_lens"],
        }
        json_results["rollouts"].append(rollout_data)

    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    # ---- Shared Visualizations (using exp_utils) ----
    x_dn = "Rollout"  # x_display_name for all shared viz functions

    print("\nCreating visualizations...")

    # Logit lens heatmaps (rows = rollouts, cols = latent positions)
    visualize_logit_lens_heatmap(
        all_results, rollout_keys, tokenizer, results_dir, x_display_name=x_dn,
    )
    visualize_logit_lens_heatmap_top5(
        all_results, rollout_keys, tokenizer, results_dir, x_display_name=x_dn,
    )

    # Full logit lens grid (one subplot per rollout)
    visualize_full_logit_lens_grid(
        all_results, rollout_keys, tokenizer, results_dir, x_display_name=x_dn,
    )

    # Accuracy fraction bar chart
    visualize_accuracy_fraction(all_rollout_results, results_dir)

    # Cosine similarity analysis
    print("\nComputing cosine similarity matrices...")
    cos_matrices = compute_cosine_similarity_matrices(all_vectors, rollout_keys)
    visualize_cosine_similarity_matrices(
        cos_matrices, rollout_keys, results_dir, x_display_name=x_dn,
    )

    print("Computing consecutive cosine similarities...")
    consec_sims = compute_consecutive_cosine_sims(all_vectors, rollout_keys)
    visualize_consecutive_cosine_sims(
        consec_sims, rollout_keys, results_dir, x_display_name=x_dn,
    )

    print("Computing drift from first rollout...")
    drift = compute_drift_from_first(all_vectors, rollout_keys)
    visualize_drift_from_first(
        drift, rollout_keys, results_dir, x_display_name=x_dn,
    )

    # Vector norms
    visualize_vector_norms(
        all_vectors, rollout_keys, results_dir, x_display_name=x_dn,
    )

    # Per-layer logit lens heatmaps
    num_ll_layers = len(
        all_results[0]["latent_logit_lens"][0]["logit_lens"]
    )
    print(f"\nGenerating per-layer logit lens heatmaps ({num_ll_layers} layers)...")
    for li in range(num_ll_layers):
        visualize_logit_lens_heatmap(
            all_results, rollout_keys, tokenizer, results_dir,
            x_display_name=x_dn, layer_index=li,
        )
        visualize_logit_lens_heatmap_top5(
            all_results, rollout_keys, tokenizer, results_dir,
            x_display_name=x_dn, layer_index=li,
        )

    # Cross-layer analysis
    print("\nCross-layer analysis...")
    visualize_token_first_occurrence_depth(
        all_results, rollout_keys, tokenizer, results_dir,
        x_display_name=x_dn,
    )
    visualize_layer_entropy(
        all_results, rollout_keys, results_dir, x_display_name=x_dn,
    )
    visualize_token_stability_across_layers(
        all_results, rollout_keys, results_dir, x_display_name=x_dn,
    )

    # Dimensionality reduction
    print("Running t-SNE...")
    visualize_tsne(
        all_vectors, rollout_keys, results_dir,
        x_display_name=x_dn, perplexity=tsne_perplexity, seed=seed,
    )

    print("Running UMAP...")
    visualize_umap(
        all_vectors, rollout_keys, results_dir,
        x_display_name=x_dn, seed=seed,
    )

    print("Generating within-series cosine similarity plots...")
    visualize_within_series_cosine_similarity(
        all_vectors, rollout_keys, results_dir,
        x_display_name=x_dn,
    )

    print("\nExperiment complete!")


# %%
if __name__ == "__main__":
    fire.Fire(main)
