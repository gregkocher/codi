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
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv

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
# Experiment-12-specific Visualizations
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

    # ---- Visualizations (shared) ----
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

    print("Generating within-series cosine similarity plots...")
    visualize_within_series_cosine_similarity(
        all_vectors, x_values, results_dir, x_display_name="X",
    )

    print("\nExperiment complete!")


# %%
if __name__ == "__main__":
    fire.Fire(main)
