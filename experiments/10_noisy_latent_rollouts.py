# ABOUTME: Noisy latent rollout experiment with logit lens analysis (batched).
# ABOUTME: Runs N rollouts on the same prompt in a single batched forward pass,
# ABOUTME: with norm-preserving noise at specified latent positions.

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

# %%
# Parameters
CHECKPOINT_PATH = "bcywinski/codi_llama1b-answer_only"
MODEL_NAME_OR_PATH = "meta-llama/Llama-3.2-1B-Instruct"

DEVICE = "cuda"
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
# Batched Rollouts with Noise + Logit Lens
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

    Args:
        logit_lens_layers: Optional list of layer indices for logit lens analysis.
            None = all layers. E.g. [-4,-3,-2,-1] for last 4 layers only.

    Returns:
        List of N dicts, each with generated_text, generated_tokens, and
        per-latent-step logit lens results.
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

    with torch.no_grad():
        # ----- Step 1: Prefill once (batch_size=1) -----
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=None,
            attention_mask=attention_mask,
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
                    {"position": i, "logit_lens": lens_result}
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
            }
        )

    return results


# ============================================================================
# Visualization
# ============================================================================


def visualize_rollout_comparison(
    all_rollout_results, tokenizer, results_dir
):
    """
    Create a grid visualization: rows = rollouts, columns = latent positions.
    Each cell shows the top-1 token from the final analyzed layer's logit lens.
    """
    num_rollouts = len(all_rollout_results)
    num_latent_positions = len(all_rollout_results[0]["latent_logit_lens"])

    # Build matrix of top-1 tokens at the final analyzed layer
    token_matrix = []
    prob_matrix = np.zeros((num_rollouts, num_latent_positions))

    for r_idx, rollout in enumerate(all_rollout_results):
        row_tokens = []
        for p_idx, pos_data in enumerate(rollout["latent_logit_lens"]):
            # Get the last analyzed layer's top-1 token
            final_layer = pos_data["logit_lens"][-1]
            top_token_id = final_layer["top_indices"][0]
            top_prob = final_layer["top_probs"][0]
            token_str = tokenizer.decode([top_token_id])
            row_tokens.append(token_str)
            prob_matrix[r_idx, p_idx] = top_prob
        token_matrix.append(row_tokens)

    # Create figure
    fig, ax = plt.subplots(figsize=(3 + num_latent_positions * 1.8, 2 + num_rollouts * 0.7))

    im = ax.imshow(prob_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Top-1 probability", rotation=270, labelpad=15, fontsize=12)

    # Annotate cells with token text
    for i in range(num_rollouts):
        for j in range(num_latent_positions):
            token = token_matrix[i][j]
            prob = prob_matrix[i, j]
            token_display = repr(token)[1:-1] if token else ""
            text_color = "white" if prob > 0.5 else "black"
            ax.text(
                j, i, token_display,
                ha="center", va="center", color=text_color, fontsize=11,
            )

    ax.set_xlabel("Latent Position", fontsize=14, fontweight="bold")
    ax.set_ylabel("Rollout", fontsize=14, fontweight="bold")
    ax.set_xticks(range(num_latent_positions))
    ax.set_xticklabels([str(i) for i in range(num_latent_positions)], fontsize=11)
    ax.set_yticks(range(num_rollouts))
    ax.set_yticklabels([str(i) for i in range(num_rollouts)], fontsize=11)
    ax.set_title(
        "Top-1 Logit Lens Token (Final Layer) Across Rollouts",
        fontsize=16, fontweight="bold", pad=12,
    )

    plt.tight_layout()
    output_path = results_dir / "logit_lens_comparison.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def visualize_full_logit_lens_grid(
    all_rollout_results, tokenizer, results_dir
):
    """
    Create a multi-panel figure: one heatmap per rollout showing
    (analyzed_layer x latent_position) with top-1 token annotations.
    """
    num_rollouts = len(all_rollout_results)
    num_positions = len(all_rollout_results[0]["latent_logit_lens"])

    # Determine the set of analyzed layer indices from the first rollout
    analyzed_layers = [
        entry["layer"]
        for entry in all_rollout_results[0]["latent_logit_lens"][0]["logit_lens"]
    ]
    num_analyzed = len(analyzed_layers)

    cols = min(num_rollouts, 5)
    rows = (num_rollouts + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 5, rows * 4),
        squeeze=False,
    )

    for r_idx, rollout in enumerate(all_rollout_results):
        ax = axes[r_idx // cols][r_idx % cols]

        prob_matrix = np.zeros((num_analyzed, num_positions))
        token_matrix = [[None] * num_positions for _ in range(num_analyzed)]

        for p_idx, pos_data in enumerate(rollout["latent_logit_lens"]):
            for row_idx, layer_data in enumerate(pos_data["logit_lens"]):
                top_token_id = layer_data["top_indices"][0]
                top_prob = layer_data["top_probs"][0]
                prob_matrix[row_idx, p_idx] = top_prob
                token_matrix[row_idx][p_idx] = tokenizer.decode([top_token_id])

        im = ax.imshow(prob_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

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

        correct_str = "correct" if rollout.get("correct", False) else "wrong"
        ax.set_title(
            f"Rollout {r_idx} ({correct_str}: {rollout.get('generated_text', '')[:20]})",
            fontsize=10,
        )
        ax.set_xlabel("Latent Pos", fontsize=9)
        ax.set_ylabel("Layer", fontsize=9)
        ax.set_xticks(range(num_positions))
        ax.set_xticklabels([str(i) for i in range(num_positions)], fontsize=8)
        ax.set_yticks(range(num_analyzed))
        ax.set_yticklabels([str(l) for l in analyzed_layers], fontsize=8)

    # Hide unused axes
    for idx in range(num_rollouts, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    plt.tight_layout()
    output_path = results_dir / "logit_lens_full_grid.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_summary_table(all_rollout_results, tokenizer):
    """Print a console summary table: top-1 token at the final analyzed layer for each rollout x latent position."""
    num_positions = len(all_rollout_results[0]["latent_logit_lens"])

    header = f"{'Rollout':>8s}"
    for p in range(num_positions):
        header += f" | {'Pos ' + str(p):>12s}"
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

    # Save results to JSON
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

    for r in all_rollout_results:
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

    # Visualizations
    print("\nCreating visualizations...")
    visualize_rollout_comparison(all_rollout_results, tokenizer, results_dir)
    visualize_full_logit_lens_grid(all_rollout_results, tokenizer, results_dir)

    print("\nExperiment complete!")


# %%
if __name__ == "__main__":
    fire.Fire(main)
