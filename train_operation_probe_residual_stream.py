# ABOUTME: Trains linear probes on residual stream at each layer to detect mathematical operations.
# ABOUTME: Evaluates probe accuracy per layer for each latent position.

# %%
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from src.model import CODI

load_dotenv()

# %%
# Configuration
CHECKPOINT_PATH = "bcywinski/codi_llama1b-answer_only"
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
NUM_LATENT = 6
USE_PRJ = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Number variations to generate training/test data
NUM_VARIATIONS = 50
RANDOM_SEED = 42
TEST_RANDOM_SEED = 43

# Output paths
OUTPUT_DIR = Path("results/operation_probe_residual_stream")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Prompt templates

# Training template (for computing the probe direction)
TRAINING_TEMPLATE = (
    "A team starts with {X} members. {Y} members leave the team. "
    "Then each current member recruits {Z} additional people. "
    "How many people are there now on the team? Give the answer only and nothing else."
)

# Test templates for ADDITION (first operation is addition)
ADDITION_TEMPLATES = [
    "A company starts with {X} employees. They hire {Y} more employees. Then each current employee brings in {Z} additional people. How many people are there now in the company? Give the answer only and nothing else.",
    "A school starts with {X} students. They enroll {Y} new students. Then each current student brings {Z} additional students. How many students are there now in the school? Give the answer only and nothing else.",
    "A club starts with {X} members. They add {Y} new members. Then each current member invites {Z} additional people. How many people are there now in the club? Give the answer only and nothing else.",
    "A restaurant starts with {X} customers. They welcome {Y} more customers. Then each current customer brings {Z} additional customers. How many customers are there now in the restaurant? Give the answer only and nothing else.",
    "A gym starts with {X} members. They sign up {Y} new members. Then each current member refers {Z} additional people. How many people are there now in the gym? Give the answer only and nothing else.",
    "A band starts with {X} musicians. They add {Y} more musicians. Then each current musician brings {Z} additional musicians. How many musicians are there now in the band? Give the answer only and nothing else.",
    "A community starts with {X} residents. They welcome {Y} new residents. Then each current resident invites {Z} additional people. How many people are there now in the community? Give the answer only and nothing else.",
    "A group starts with {X} participants. They add {Y} new participants. Then each current participant brings {Z} additional people. How many people are there now in the group? Give the answer only and nothing else.",
    "A workshop starts with {X} attendees. They register {Y} more attendees. Then each current attendee brings {Z} additional people. How many people are there now in the workshop? Give the answer only and nothing else.",
]

# Test templates for SUBTRACTION (first operation is subtraction)
SUBTRACTION_TEMPLATES = [
    "A company starts with {X} employees. {Y} employees quit. Then each remaining employee brings in {Z} additional people. How many people are there now in the company? Give the answer only and nothing else.",
    "A school starts with {X} students. {Y} students transfer out. Then each remaining student brings {Z} additional students. How many students are there now in the school? Give the answer only and nothing else.",
    "A club starts with {X} members. {Y} members resign. Then each remaining member invites {Z} additional people. How many people are there now in the club? Give the answer only and nothing else.",
    "A restaurant starts with {X} customers. {Y} customers leave. Then each remaining customer brings {Z} additional customers. How many customers are there now in the restaurant? Give the answer only and nothing else.",
    "A gym starts with {X} members. {Y} members cancel their membership. Then each remaining member refers {Z} additional people. How many people are there now in the gym? Give the answer only and nothing else.",
    "A band starts with {X} musicians. {Y} musicians quit. Then each remaining musician brings {Z} additional musicians. How many musicians are there now in the band? Give the answer only and nothing else.",
    "A community starts with {X} residents. {Y} residents move away. Then each remaining resident invites {Z} additional people. How many people are there now in the community? Give the answer only and nothing else.",
    "A group starts with {X} participants. {Y} participants drop out. Then each remaining participant brings {Z} additional people. How many people are there now in the group? Give the answer only and nothing else.",
    "A workshop starts with {X} attendees. {Y} attendees leave early. Then each remaining attendee brings {Z} additional people. How many people are there now in the workshop? Give the answer only and nothing else.",
]


# %%
def generate_number_variations(
    n_variations: int, seed: int
) -> list[tuple[int, int, int]]:
    """Generate random (X, Y, Z) number tuples."""
    random.seed(seed)
    variations = []
    for _ in range(n_variations):
        # X: 5-20, Y: 2-10 (Y < X for subtraction to work), Z: 1-5
        x = random.randint(5, 20)
        y = random.randint(2, min(x - 1, 10))
        z = random.randint(1, 5)
        variations.append((x, y, z))
    return variations


# %%
def extract_residual_stream_per_layer(
    model,
    tokenizer,
    prompt: str,
    num_latent_iterations: int,
    sot_token: int,
    eot_token: int,
) -> dict[int, list[torch.Tensor]]:
    """
    Extract residual stream (hidden states) at each layer for each latent position.

    Returns:
        dict mapping latent_position -> list of tensors (one per layer)
        Each tensor has shape [hidden_dim]
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    # Add BOT token
    bot_tensor = torch.tensor(
        [[tokenizer.eos_token_id, sot_token]], dtype=torch.long, device=DEVICE
    )
    input_ids_bot = torch.cat((input_ids, bot_tensor), dim=1)
    attention_mask_bot = torch.cat(
        (attention_mask, torch.ones_like(bot_tensor, device=DEVICE)), dim=1
    )

    residual_streams = {}  # latent_position -> [layer0_hidden, layer1_hidden, ...]

    with torch.no_grad():
        # Prefill phase
        outputs = model.codi(
            input_ids=input_ids_bot,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=None,
            attention_mask=attention_mask_bot,
        )
        past_key_values = outputs.past_key_values

        # Initial latent embedding (position 0)
        # hidden_states is tuple of (num_layers + 1) tensors, each [batch, seq_len, hidden]
        # We want the last token position from each layer
        layer_hiddens = []
        for layer_idx, hs in enumerate(outputs.hidden_states):
            layer_hiddens.append(hs[:, -1, :].squeeze(0).cpu())  # [hidden_dim]
        residual_streams[0] = layer_hiddens

        # Get latent embedding for next iteration
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        if model.use_prj:
            latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        # Latent iterations (positions 1 to num_latent_iterations)
        for i in range(num_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values

            # Extract hidden states from each layer
            layer_hiddens = []
            for layer_idx, hs in enumerate(outputs.hidden_states):
                layer_hiddens.append(hs[:, -1, :].squeeze(0).cpu())  # [hidden_dim]
            residual_streams[i + 1] = layer_hiddens

            # Prepare for next iteration
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

    return residual_streams


# %%
def compute_averaged_probes_per_layer(
    model,
    tokenizer,
    template: str,
    variations: list[tuple[int, int, int]],
    latent_position: int,
    sot_token: int,
    eot_token: int,
    num_layers: int,
) -> list[torch.Tensor]:
    """
    Compute averaged probe vectors for each layer at a specific latent position.

    Returns:
        List of probe tensors (one per layer), each normalized to unit norm.
    """
    # Collect hidden states per layer
    layer_vectors = {layer_idx: [] for layer_idx in range(num_layers)}

    for x, y, z in tqdm(
        variations, desc=f"Computing probes for latent pos {latent_position}"
    ):
        prompt = template.format(X=x, Y=y, Z=z)
        residual_streams = extract_residual_stream_per_layer(
            model, tokenizer, prompt, NUM_LATENT, sot_token, eot_token
        )

        # Get hidden states for the target latent position
        layer_hiddens = residual_streams[latent_position]
        for layer_idx, hidden in enumerate(layer_hiddens):
            layer_vectors[layer_idx].append(hidden)

    # Average and normalize for each layer
    probes = []
    for layer_idx in range(num_layers):
        stacked = torch.stack(
            layer_vectors[layer_idx], dim=0
        )  # [n_variations, hidden_dim]
        averaged = stacked.mean(dim=0)  # [hidden_dim]
        probe = averaged / averaged.norm()  # Normalize to unit vector
        probes.append(probe)

    return probes


# %%
def evaluate_probes_on_templates(
    model,
    tokenizer,
    probes: list[torch.Tensor],
    templates: list[str],
    variations: list[tuple[int, int, int]],
    latent_position: int,
    sot_token: int,
    eot_token: int,
) -> dict:
    """
    Evaluate probes (one per layer) on test templates.

    Returns:
        dict with per-layer statistics
    """
    num_layers = len(probes)
    layer_dot_products = {layer_idx: [] for layer_idx in range(num_layers)}

    for template in tqdm(templates, desc="Evaluating templates"):
        for x, y, z in variations:
            prompt = template.format(X=x, Y=y, Z=z)
            residual_streams = extract_residual_stream_per_layer(
                model, tokenizer, prompt, NUM_LATENT, sot_token, eot_token
            )

            layer_hiddens = residual_streams[latent_position]
            for layer_idx, hidden in enumerate(layer_hiddens):
                # Normalize test vector
                hidden_norm = hidden / hidden.norm()
                # Compute dot product with probe
                dot_product = torch.dot(probes[layer_idx], hidden_norm).item()
                layer_dot_products[layer_idx].append(dot_product)

    # Compute statistics per layer
    results = {
        "per_layer": [],
        "means": [],
        "stds": [],
    }
    for layer_idx in range(num_layers):
        mean = np.mean(layer_dot_products[layer_idx])
        std = np.std(layer_dot_products[layer_idx])
        results["per_layer"].append(
            {
                "layer": layer_idx,
                "mean": mean,
                "std": std,
                "dot_products": layer_dot_products[layer_idx],
            }
        )
        results["means"].append(mean)
        results["stds"].append(std)

    return results


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
EOT_TOKEN = tokenizer.eocot_id

# Get number of layers
NUM_LAYERS = model.codi.config.num_hidden_layers + 1  # +1 for embedding layer
print(f"Number of layers (including embedding): {NUM_LAYERS}")

# %%
print("\nGenerating number variations...")
train_variations = generate_number_variations(NUM_VARIATIONS, RANDOM_SEED)
test_variations = generate_number_variations(NUM_VARIATIONS, TEST_RANDOM_SEED)
print(f"Generated {len(train_variations)} training variations")
print(f"Generated {len(test_variations)} test variations")
print(f"Sample training variations: {train_variations[:5]}")
print(f"Sample test variations: {test_variations[:5]}")

# %%
# Store all results
all_results = {}

# Process each latent position
for latent_pos in range(NUM_LATENT + 1):  # 0 to NUM_LATENT inclusive
    print(f"\n{'=' * 80}")
    print(f"Processing Latent Position {latent_pos}")
    print(f"{'=' * 80}")

    # Train probes for each layer
    print(f"\nTraining probes on residual stream at latent position {latent_pos}...")
    probes = compute_averaged_probes_per_layer(
        model,
        tokenizer,
        TRAINING_TEMPLATE,
        train_variations,
        latent_pos,
        SOT_TOKEN,
        EOT_TOKEN,
        NUM_LAYERS,
    )
    print(f"Trained {len(probes)} probes (one per layer)")

    # Evaluate on addition templates
    print("\nEvaluating probes on ADDITION templates...")
    addition_results = evaluate_probes_on_templates(
        model,
        tokenizer,
        probes,
        ADDITION_TEMPLATES,
        test_variations,
        latent_pos,
        SOT_TOKEN,
        EOT_TOKEN,
    )

    # Evaluate on subtraction templates
    print("\nEvaluating probes on SUBTRACTION templates...")
    subtraction_results = evaluate_probes_on_templates(
        model,
        tokenizer,
        probes,
        SUBTRACTION_TEMPLATES,
        test_variations,
        latent_pos,
        SOT_TOKEN,
        EOT_TOKEN,
    )

    all_results[latent_pos] = {
        "addition": addition_results,
        "subtraction": subtraction_results,
    }

    # Create plot for this latent position
    fig, ax = plt.subplots(figsize=(12, 6))

    layers = list(range(NUM_LAYERS))
    add_means = addition_results["means"]
    add_stds = addition_results["stds"]
    sub_means = subtraction_results["means"]
    sub_stds = subtraction_results["stds"]

    ax.errorbar(
        layers,
        add_means,
        yerr=add_stds,
        label="Addition",
        color="blue",
        marker="o",
        capsize=3,
        alpha=0.7,
        linewidth=2,
        markersize=4,
    )
    ax.errorbar(
        layers,
        sub_means,
        yerr=sub_stds,
        label="Subtraction",
        color="red",
        marker="s",
        capsize=3,
        alpha=0.7,
        linewidth=2,
        markersize=4,
    )

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Dot Product with Probe", fontsize=12)
    ax.set_title(
        f"Operation Probe Accuracy by Layer (Latent Position {latent_pos})", fontsize=14
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers[::2])  # Show every other layer for readability

    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"probe_accuracy_latent_pos_{latent_pos}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_path}")

    # Print summary for this latent position
    print(f"\nSummary for Latent Position {latent_pos}:")
    print(f"  Addition - Mean across layers: {np.mean(add_means):.4f}")
    print(f"  Subtraction - Mean across layers: {np.mean(sub_means):.4f}")
    print(f"  Difference: {np.mean(add_means) - np.mean(sub_means):.4f}")

# %%
# Save all results to JSON
results_file = OUTPUT_DIR / "all_results.json"
# Convert numpy arrays to lists for JSON serialization
json_results = {}
for latent_pos, res in all_results.items():
    json_results[str(latent_pos)] = {
        "addition": {
            "means": res["addition"]["means"],
            "stds": res["addition"]["stds"],
        },
        "subtraction": {
            "means": res["subtraction"]["means"],
            "stds": res["subtraction"]["stds"],
        },
    }

with open(results_file, "w") as f:
    json.dump(json_results, f, indent=2)
print(f"\nAll results saved to {results_file}")

# %%
# Create summary plot with all latent positions
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for latent_pos in range(NUM_LATENT + 1):
    ax = axes[latent_pos]
    res = all_results[latent_pos]

    layers = list(range(NUM_LAYERS))
    add_means = res["addition"]["means"]
    sub_means = res["subtraction"]["means"]

    ax.plot(layers, add_means, label="Addition", color="blue", linewidth=2, alpha=0.7)
    ax.plot(layers, sub_means, label="Subtraction", color="red", linewidth=2, alpha=0.7)
    ax.fill_between(
        layers,
        np.array(add_means) - np.array(res["addition"]["stds"]),
        np.array(add_means) + np.array(res["addition"]["stds"]),
        color="blue",
        alpha=0.2,
    )
    ax.fill_between(
        layers,
        np.array(sub_means) - np.array(res["subtraction"]["stds"]),
        np.array(sub_means) + np.array(res["subtraction"]["stds"]),
        color="red",
        alpha=0.2,
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Dot Product")
    ax.set_title(f"Latent Position {latent_pos}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Hide unused subplot
if NUM_LATENT + 1 < len(axes):
    for i in range(NUM_LATENT + 1, len(axes)):
        axes[i].axis("off")

plt.suptitle(
    "Operation Probe Accuracy by Layer for Each Latent Position", fontsize=16, y=1.02
)
plt.tight_layout()
summary_plot_path = OUTPUT_DIR / "summary_all_latent_positions.png"
plt.savefig(summary_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSummary plot saved to {summary_plot_path}")

# %%
# Print final summary
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"\nNumber of layers: {NUM_LAYERS}")
print(f"Number of latent positions: {NUM_LATENT + 1}")
print(f"Number of training variations: {len(train_variations)}")
print(f"Number of test variations: {len(test_variations)}")
print(f"\nTraining template: {TRAINING_TEMPLATE[:60]}...")
print(f"Addition test templates: {len(ADDITION_TEMPLATES)}")
print(f"Subtraction test templates: {len(SUBTRACTION_TEMPLATES)}")

print("\nMean difference (Addition - Subtraction) per latent position:")
for latent_pos in range(NUM_LATENT + 1):
    res = all_results[latent_pos]
    add_mean = np.mean(res["addition"]["means"])
    sub_mean = np.mean(res["subtraction"]["means"])
    diff = add_mean - sub_mean
    print(f"  Latent Position {latent_pos}: {diff:.4f}")

print("=" * 80)
