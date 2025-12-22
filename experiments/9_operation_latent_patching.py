# ABOUTME: Tests causal effect of latent vectors by patching addition latents into subtraction prompts.
# ABOUTME: Uses logit lens at position 2 to compare (X+Y) vs (X-Y) probabilities with/without patching.

# %%
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import CODI

load_dotenv()

# %%
# Configuration
CHECKPOINT_PATH = "bcywinski/codi_llama1b-answer_only"
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
NUM_LATENT = 6
USE_PRJ = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Experiment parameters
LATENT_POSITION_TO_PROBE = 2  # Position where (X+Y) or (X-Y) is represented
NUM_LATENTS_TO_PATCH = 3  # Patch latent positions 0, 1, 2
MAX_PROMPTS = 100  # Limit prompts for faster experimentation

# Paths
PROMPTS_PATH = Path(__file__).parent.parent / "prompts" / "prompts.json"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "operation_latent_patching"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %%
def load_prompts(prompts_path: Path) -> list[dict]:
    """Load prompts from JSON file."""
    with open(prompts_path, "r") as f:
        data = json.load(f)
    return data["prompts"]


# %%
def get_lm_head(model):
    """Get the language model head (unembedding matrix) from the model."""
    codi = model.codi
    if hasattr(codi, "get_base_model"):
        return codi.get_base_model().lm_head
    return codi.lm_head


def get_layer_norm(model):
    """Get the final layer norm before the lm_head."""
    codi = model.codi
    if hasattr(codi, "get_base_model"):
        base = codi.get_base_model()
    else:
        base = codi

    if hasattr(base, "model") and hasattr(base.model, "norm"):
        return base.model.norm
    if hasattr(base, "transformer") and hasattr(base.transformer, "ln_f"):
        return base.transformer.ln_f
    return None


# %%
def run_with_latent_collection(
    model,
    tokenizer,
    prompt: str,
    num_latent_iterations: int,
    sot_token: int,
    device: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Run inference and collect latent embeddings and final hidden states at each position.

    Returns:
        latent_embeds: List of latent embeddings (after projection) at each position
        hidden_states: List of last-layer hidden states at each position
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    bot_tensor = torch.tensor(
        [tokenizer.eos_token_id, sot_token], dtype=torch.long, device=device
    ).unsqueeze(0)

    input_ids_bot = torch.cat((input_ids, bot_tensor), dim=1)
    attention_mask_bot = torch.cat((attention_mask, torch.ones_like(bot_tensor)), dim=1)

    latent_embeds = []
    hidden_states_list = []

    with torch.no_grad():
        # Encode prompt
        outputs = model.codi(
            input_ids=input_ids_bot,
            use_cache=True,
            output_hidden_states=True,
            attention_mask=attention_mask_bot,
        )
        past_key_values = outputs.past_key_values

        # Get initial latent embedding (position 0)
        h = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        hidden_states_list.append(h.clone())

        latent_embd = h.clone()
        if model.use_prj:
            latent_embd = model.prj(latent_embd)
            latent_embd = latent_embd.to(dtype=model.codi.dtype)
        latent_embeds.append(latent_embd.clone())

        # Latent iterations
        for i in range(num_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values

            h = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            hidden_states_list.append(h.clone())

            latent_embd = h.clone()
            if model.use_prj:
                latent_embd = model.prj(latent_embd)
                latent_embd = latent_embd.to(dtype=model.codi.dtype)
            latent_embeds.append(latent_embd.clone())

    return latent_embeds, hidden_states_list


# %%
def run_with_patched_latents_and_kv(
    model,
    tokenizer,
    prompt: str,
    num_latent_iterations: int,
    sot_token: int,
    device: str,
    source_latents: list[torch.Tensor],
    source_hidden_states: list[torch.Tensor],
    num_positions_to_patch: int,
) -> list[torch.Tensor]:
    """
    Run inference patching both latent embeddings AND hidden states from source.

    For positions < num_positions_to_patch, we use the source hidden state directly
    (bypassing the transformer forward pass for those positions) and then continue
    normally from the last patched position.

    Returns:
        hidden_states: List of last-layer hidden states at each position
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    bot_tensor = torch.tensor(
        [tokenizer.eos_token_id, sot_token], dtype=torch.long, device=device
    ).unsqueeze(0)

    input_ids_bot = torch.cat((input_ids, bot_tensor), dim=1)
    attention_mask_bot = torch.cat((attention_mask, torch.ones_like(bot_tensor)), dim=1)

    hidden_states_list = []

    with torch.no_grad():
        # Process the target prompt to get its KV cache
        outputs = model.codi(
            input_ids=input_ids_bot,
            use_cache=True,
            output_hidden_states=True,
            attention_mask=attention_mask_bot,
        )
        past_key_values = outputs.past_key_values

        # For the patched positions, we use source hidden states directly
        for pos in range(num_positions_to_patch):
            if pos < len(source_hidden_states):
                hidden_states_list.append(source_hidden_states[pos].clone())

        # Use the last patched latent embedding to continue
        last_patched_pos = min(num_positions_to_patch - 1, len(source_latents) - 1)
        latent_embd = source_latents[last_patched_pos].clone()

        # Continue from the patched position
        for i in range(num_positions_to_patch - 1, num_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values

            h = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            # Only append if we're past the patched positions
            if i >= num_positions_to_patch - 1:
                hidden_states_list.append(h.clone())

            latent_embd = h.clone()
            if model.use_prj:
                latent_embd = model.prj(latent_embd)
                latent_embd = latent_embd.to(dtype=model.codi.dtype)

    return hidden_states_list


# %%
def get_token_probability(
    hidden_state: torch.Tensor,
    lm_head,
    layer_norm,
    token_id: int,
) -> float:
    """
    Apply logit lens to get probability of a specific token.

    Args:
        hidden_state: Shape (1, 1, hidden_dim)
        lm_head: Language model head
        layer_norm: Final layer norm
        token_id: Token ID to get probability for

    Returns:
        Probability of the token
    """
    h = hidden_state.squeeze(0).squeeze(0)  # (hidden_dim,)

    if layer_norm is not None:
        h = layer_norm(h.unsqueeze(0)).squeeze(0)

    logits = lm_head(h.unsqueeze(0))  # (1, vocab)
    probs = torch.softmax(logits, dim=-1)

    return probs[0, token_id].item()


def get_number_token_id(tokenizer, number: int) -> int:
    """Get the token ID for a number."""
    # Try different formats
    for fmt in [str(number), f" {number}", f"{number}"]:
        tokens = tokenizer.encode(fmt, add_special_tokens=False)
        if len(tokens) == 1:
            return tokens[0]
    # Fallback: just use the first token
    return tokenizer.encode(str(number), add_special_tokens=False)[0]


# %%
print("Loading CODI model...")
model = CODI.from_pretrained(
    checkpoint_path=CHECKPOINT_PATH,
    model_name_or_path=BASE_MODEL,
    lora_r=128,
    lora_alpha=32,
    num_latent=NUM_LATENT,
    use_prj=USE_PRJ,
    device=DEVICE,
    dtype="bfloat16",
    strict=False,
    remove_eos=False,
    full_precision=True,
)
model.eval()
print(f"Model loaded on {DEVICE}")

# Configure tokenizer with special tokens
tokenizer = model.tokenizer
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token_id = model.pad_token_id
tokenizer.add_special_tokens({"additional_special_tokens": ["<|bocot|>", "<|eocot|>"]})
tokenizer.bocot_id = tokenizer.convert_tokens_to_ids("<|bocot|>")
tokenizer.eocot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")
SOT_TOKEN = tokenizer.bocot_id

lm_head = get_lm_head(model)
layer_norm = get_layer_norm(model)

# %%
print("\nLoading prompts...")
prompts = load_prompts(PROMPTS_PATH)
prompts = prompts[:MAX_PROMPTS]
print(f"Using {len(prompts)} prompts")

# %%
print(f"\nRunning experiment: patching first {NUM_LATENTS_TO_PATCH} latent positions")
print(f"Probing at latent position {LATENT_POSITION_TO_PROBE}")

results = []

for prompt_data in tqdm(prompts, desc="Processing prompts"):
    X = prompt_data["X"]
    Y = prompt_data["Y"]
    add_result = X + Y
    sub_result = X - Y

    add_prompt = prompt_data["addition"]["prompt"]
    sub_prompt = prompt_data["subtraction"]["prompt"]

    # Get token IDs for the intermediate results
    add_token_id = get_number_token_id(tokenizer, add_result)
    sub_token_id = get_number_token_id(tokenizer, sub_result)

    # Run addition prompt and collect latent embeddings and hidden states
    add_latents, add_hidden = run_with_latent_collection(
        model, tokenizer, add_prompt, NUM_LATENT, SOT_TOKEN, DEVICE
    )

    # Run subtraction prompt normally (baseline)
    _, sub_hidden_baseline = run_with_latent_collection(
        model, tokenizer, sub_prompt, NUM_LATENT, SOT_TOKEN, DEVICE
    )

    # Run subtraction prompt with patched latents AND hidden states from addition
    sub_hidden_patched = run_with_patched_latents_and_kv(
        model,
        tokenizer,
        sub_prompt,
        NUM_LATENT,
        SOT_TOKEN,
        DEVICE,
        add_latents,
        add_hidden,
        NUM_LATENTS_TO_PATCH,
    )

    # Get probabilities at the probe position
    probe_pos = LATENT_POSITION_TO_PROBE

    # Baseline subtraction: P(X+Y) and P(X-Y)
    baseline_add_prob = get_token_probability(
        sub_hidden_baseline[probe_pos], lm_head, layer_norm, add_token_id
    )
    baseline_sub_prob = get_token_probability(
        sub_hidden_baseline[probe_pos], lm_head, layer_norm, sub_token_id
    )

    # Patched subtraction: P(X+Y) and P(X-Y)
    # Note: probe_pos should be within the patched hidden states
    if probe_pos < len(sub_hidden_patched):
        patched_add_prob = get_token_probability(
            sub_hidden_patched[probe_pos], lm_head, layer_norm, add_token_id
        )
        patched_sub_prob = get_token_probability(
            sub_hidden_patched[probe_pos], lm_head, layer_norm, sub_token_id
        )
    else:
        patched_add_prob = 0.0
        patched_sub_prob = 0.0

    # Addition baseline (sanity check): P(X+Y) at the addition prompt
    addition_add_prob = get_token_probability(
        add_hidden[probe_pos], lm_head, layer_norm, add_token_id
    )
    addition_sub_prob = get_token_probability(
        add_hidden[probe_pos], lm_head, layer_norm, sub_token_id
    )

    results.append(
        {
            "X": X,
            "Y": Y,
            "add_result": add_result,
            "sub_result": sub_result,
            "baseline_add_prob": baseline_add_prob,
            "baseline_sub_prob": baseline_sub_prob,
            "patched_add_prob": patched_add_prob,
            "patched_sub_prob": patched_sub_prob,
            "addition_add_prob": addition_add_prob,
            "addition_sub_prob": addition_sub_prob,
        }
    )

# %%
# Compute averages
baseline_add_probs = [r["baseline_add_prob"] for r in results]
baseline_sub_probs = [r["baseline_sub_prob"] for r in results]
patched_add_probs = [r["patched_add_prob"] for r in results]
patched_sub_probs = [r["patched_sub_prob"] for r in results]
addition_add_probs = [r["addition_add_prob"] for r in results]
addition_sub_probs = [r["addition_sub_prob"] for r in results]

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"\nLatent position probed: {LATENT_POSITION_TO_PROBE}")
print(f"Number of latent positions patched: {NUM_LATENTS_TO_PATCH}")
print(f"Number of prompts: {len(results)}")

print("\n--- Baseline (subtraction prompt, no patching) ---")
print(f"  P(X+Y): {np.mean(baseline_add_probs):.4f} +/- {np.std(baseline_add_probs):.4f}")
print(f"  P(X-Y): {np.mean(baseline_sub_probs):.4f} +/- {np.std(baseline_sub_probs):.4f}")

print("\n--- Patched (subtraction prompt, addition hidden states patched) ---")
print(f"  P(X+Y): {np.mean(patched_add_probs):.4f} +/- {np.std(patched_add_probs):.4f}")
print(f"  P(X-Y): {np.mean(patched_sub_probs):.4f} +/- {np.std(patched_sub_probs):.4f}")

print("\n--- Addition prompt (sanity check) ---")
print(f"  P(X+Y): {np.mean(addition_add_probs):.4f} +/- {np.std(addition_add_probs):.4f}")
print(f"  P(X-Y): {np.mean(addition_sub_probs):.4f} +/- {np.std(addition_sub_probs):.4f}")

# %%
# Save results
results_data = {
    "config": {
        "checkpoint_path": CHECKPOINT_PATH,
        "base_model": BASE_MODEL,
        "num_latent": NUM_LATENT,
        "latent_position_probed": LATENT_POSITION_TO_PROBE,
        "num_positions_patched": NUM_LATENTS_TO_PATCH,
        "num_prompts": len(results),
    },
    "summary": {
        "baseline_add_prob_mean": float(np.mean(baseline_add_probs)),
        "baseline_add_prob_std": float(np.std(baseline_add_probs)),
        "baseline_sub_prob_mean": float(np.mean(baseline_sub_probs)),
        "baseline_sub_prob_std": float(np.std(baseline_sub_probs)),
        "patched_add_prob_mean": float(np.mean(patched_add_probs)),
        "patched_add_prob_std": float(np.std(patched_add_probs)),
        "patched_sub_prob_mean": float(np.mean(patched_sub_probs)),
        "patched_sub_prob_std": float(np.std(patched_sub_probs)),
        "addition_add_prob_mean": float(np.mean(addition_add_probs)),
        "addition_add_prob_std": float(np.std(addition_add_probs)),
        "addition_sub_prob_mean": float(np.mean(addition_sub_probs)),
        "addition_sub_prob_std": float(np.std(addition_sub_probs)),
    },
    "per_prompt_results": results,
}

results_file = OUTPUT_DIR / "patching_results.json"
with open(results_file, "w") as f:
    json.dump(results_data, f, indent=2)
print(f"\nSaved results to {results_file}")

# %%
# Plot comparison
fontsize_title = 20
fontsize_label = 18
fontsize_tick = 16

fig, ax = plt.subplots(figsize=(10, 6))

categories = ["Baseline\n(subtraction)", "Patched\n(addition latents)", "Addition\n(sanity check)"]
x = np.arange(len(categories))
width = 0.35

# Data for plotting
add_means = [
    np.mean(baseline_add_probs),
    np.mean(patched_add_probs),
    np.mean(addition_add_probs),
]
add_stds = [
    np.std(baseline_add_probs),
    np.std(patched_add_probs),
    np.std(addition_add_probs),
]

sub_means = [
    np.mean(baseline_sub_probs),
    np.mean(patched_sub_probs),
    np.mean(addition_sub_probs),
]
sub_stds = [
    np.std(baseline_sub_probs),
    np.std(patched_sub_probs),
    np.std(addition_sub_probs),
]

bars1 = ax.bar(
    x - width / 2,
    add_means,
    width,
    yerr=add_stds,
    label="P(X+Y)",
    color="#3498db",
    capsize=5,
    edgecolor="black",
    linewidth=1.5,
)
bars2 = ax.bar(
    x + width / 2,
    sub_means,
    width,
    yerr=sub_stds,
    label="P(X-Y)",
    color="#e74c3c",
    capsize=5,
    edgecolor="black",
    linewidth=1.5,
)

ax.set_ylabel("Probability", fontsize=fontsize_label, fontweight="bold")
ax.set_title(
    f"Logit Lens at Latent Position {LATENT_POSITION_TO_PROBE}\n(Patching first {NUM_LATENTS_TO_PATCH} latent positions)",
    fontsize=fontsize_title,
    fontweight="bold",
)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=fontsize_tick)
ax.tick_params(axis="y", labelsize=fontsize_tick)
ax.legend(fontsize=fontsize_tick)
ax.set_ylim(0, max(max(add_means), max(sub_means)) * 1.3)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plot_path = OUTPUT_DIR / "patching_comparison.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved plot to {plot_path}")

# %%
# Additional plot: distribution comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: P(X+Y) distributions
ax1 = axes[0]
ax1.hist(baseline_add_probs, bins=20, alpha=0.5, label="Baseline (sub)", color="#e74c3c")
ax1.hist(patched_add_probs, bins=20, alpha=0.5, label="Patched", color="#3498db")
ax1.hist(addition_add_probs, bins=20, alpha=0.5, label="Addition", color="#2ecc71")
ax1.set_xlabel("P(X+Y)", fontsize=fontsize_label, fontweight="bold")
ax1.set_ylabel("Count", fontsize=fontsize_label, fontweight="bold")
ax1.set_title("Distribution of P(X+Y)", fontsize=fontsize_title, fontweight="bold")
ax1.legend(fontsize=12)
ax1.tick_params(labelsize=fontsize_tick)

# Right: P(X-Y) distributions
ax2 = axes[1]
ax2.hist(baseline_sub_probs, bins=20, alpha=0.5, label="Baseline (sub)", color="#e74c3c")
ax2.hist(patched_sub_probs, bins=20, alpha=0.5, label="Patched", color="#3498db")
ax2.set_xlabel("P(X-Y)", fontsize=fontsize_label, fontweight="bold")
ax2.set_ylabel("Count", fontsize=fontsize_label, fontweight="bold")
ax2.set_title("Distribution of P(X-Y)", fontsize=fontsize_title, fontweight="bold")
ax2.legend(fontsize=12)
ax2.tick_params(labelsize=fontsize_tick)

plt.tight_layout()
dist_plot_path = OUTPUT_DIR / "probability_distributions.png"
plt.savefig(dist_plot_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved distribution plot to {dist_plot_path}")

# %%
