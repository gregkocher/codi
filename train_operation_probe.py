# ABOUTME: Trains a linear probe to detect mathematical operations (addition vs subtraction)
# ABOUTME: in CODI latent vectors and evaluates via dot product on held-out prompts.

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

# Latent position to analyze (0 = initial, 1-6 = after each iteration)
LATENT_POSITION = 1

# Number variations to generate training/test data
NUM_VARIATIONS = 50
RANDOM_SEED = 42

# Output paths
OUTPUT_DIR = Path("results/operation_probe")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Prompt templates

# Training template (for computing the probe direction)
TRAINING_TEMPLATE = (
    "A team starts with {X} members. They recruit {Y} new members. "
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
    "A team starts with {X} members. {Y} members leave the team. Then each remaining member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else.",
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
def extract_latent_vector(
    model, tokenizer, prompt: str, latent_position: int, sot_token: int, eot_token: int
) -> torch.Tensor:
    """Extract latent vector at specified position for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            tokenizer=tokenizer,
            return_latent_vectors=True,
            num_latent_iterations=NUM_LATENT,
            max_new_tokens=10,
            greedy=True,
            sot_token=sot_token,
            eot_token=eot_token,
            output_hidden_states=True,
        )

    latent_vectors = output["latent_vectors"]
    # latent_vectors is a list of tensors, each shape [batch, 1, hidden_dim]
    return latent_vectors[latent_position].squeeze()  # [hidden_dim]


# %%
def compute_averaged_probe(
    model,
    tokenizer,
    template: str,
    variations: list[tuple[int, int, int]],
    latent_position: int,
    sot_token: int,
    eot_token: int,
) -> torch.Tensor:
    """Compute averaged latent vector across number variations."""
    all_vectors = []

    for x, y, z in tqdm(variations, desc="Computing probe vectors"):
        prompt = template.format(X=x, Y=y, Z=z)
        latent_vec = extract_latent_vector(
            model, tokenizer, prompt, latent_position, sot_token, eot_token
        )
        all_vectors.append(latent_vec.cpu())

    # Stack and average
    stacked = torch.stack(all_vectors, dim=0)  # [n_variations, hidden_dim]
    averaged = stacked.mean(dim=0)  # [hidden_dim]

    # Normalize to unit vector for dot product comparisons
    probe = averaged / averaged.norm()

    return probe


# %%
def evaluate_probe_on_templates(
    model,
    tokenizer,
    probe: torch.Tensor,
    templates: list[str],
    variations: list[tuple[int, int, int]],
    latent_position: int,
    sot_token: int,
    eot_token: int,
) -> dict:
    """Evaluate probe via dot product on test templates."""
    results = {
        "per_template": [],
        "all_dot_products": [],
    }

    probe = probe.to(DEVICE)

    for template_idx, template in enumerate(
        tqdm(templates, desc="Evaluating templates")
    ):
        template_dot_products = []

        for x, y, z in variations:
            prompt = template.format(X=x, Y=y, Z=z)
            latent_vec = extract_latent_vector(
                model, tokenizer, prompt, latent_position, sot_token, eot_token
            )

            # Normalize test vector too
            latent_vec_norm = latent_vec / latent_vec.norm()

            # Compute dot product
            dot_product = torch.dot(probe, latent_vec_norm).item()
            template_dot_products.append(dot_product)

        results["per_template"].append(
            {
                "template_idx": template_idx,
                "template": template[:80] + "...",
                "dot_products": template_dot_products,
                "mean": np.mean(template_dot_products),
                "std": np.std(template_dot_products),
            }
        )
        results["all_dot_products"].extend(template_dot_products)

    results["overall_mean"] = np.mean(results["all_dot_products"])
    results["overall_std"] = np.std(results["all_dot_products"])

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
    remove_eos=True,
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

# %%
print("\nGenerating number variations...")
variations = generate_number_variations(NUM_VARIATIONS, RANDOM_SEED)
print(f"Generated {len(variations)} variations")
print(f"Sample variations: {variations[:5]}")

# %%
print(f"\nTraining probe on latent position {LATENT_POSITION}...")
print(f"Using template: {TRAINING_TEMPLATE[:60]}...")
probe = compute_averaged_probe(
    model,
    tokenizer,
    TRAINING_TEMPLATE,
    variations,
    LATENT_POSITION,
    SOT_TOKEN,
    EOT_TOKEN,
)
print(f"Probe shape: {probe.shape}")
print(f"Probe norm (should be 1.0): {probe.norm().item():.6f}")

# %%
print("\nEvaluating probe on ADDITION templates...")
addition_results = evaluate_probe_on_templates(
    model,
    tokenizer,
    probe,
    ADDITION_TEMPLATES,
    variations,
    LATENT_POSITION,
    SOT_TOKEN,
    EOT_TOKEN,
)

print("\nEvaluating probe on SUBTRACTION templates...")
subtraction_results = evaluate_probe_on_templates(
    model,
    tokenizer,
    probe,
    SUBTRACTION_TEMPLATES,
    variations,
    LATENT_POSITION,
    SOT_TOKEN,
    EOT_TOKEN,
)

# %%
# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Per-template results
ax1 = axes[0]
addition_means = [t["mean"] for t in addition_results["per_template"]]
addition_stds = [t["std"] for t in addition_results["per_template"]]
subtraction_means = [t["mean"] for t in subtraction_results["per_template"]]
subtraction_stds = [t["std"] for t in subtraction_results["per_template"]]

x_add = np.arange(len(addition_means))
x_sub = np.arange(len(subtraction_means)) + len(addition_means) + 1

ax1.bar(
    x_add,
    addition_means,
    yerr=addition_stds,
    capsize=3,
    label="Addition",
    color="blue",
    alpha=0.7,
)
ax1.bar(
    x_sub,
    subtraction_means,
    yerr=subtraction_stds,
    capsize=3,
    label="Subtraction",
    color="red",
    alpha=0.7,
)

ax1.set_xlabel("Template Index")
ax1.set_ylabel("Dot Product with Probe")
ax1.set_title(f"Dot Product by Template (Latent Position {LATENT_POSITION})")
ax1.legend()
ax1.axhline(y=0, color="black", linestyle="--", alpha=0.3)

# Plot 2: Overall comparison
ax2 = axes[1]
categories = ["Addition\n(test)", "Subtraction\n(test)"]
means = [addition_results["overall_mean"], subtraction_results["overall_mean"]]
stds = [addition_results["overall_std"], subtraction_results["overall_std"]]
colors = ["blue", "red"]

bars = ax2.bar(categories, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
ax2.set_ylabel("Dot Product with Probe")
ax2.set_title(f"Overall Dot Product Comparison (Latent Position {LATENT_POSITION})")
ax2.axhline(y=0, color="black", linestyle="--", alpha=0.3)

# Add value labels on bars
for bar, mean, std in zip(bars, means, stds):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + std + 0.01,
        f"{mean:.3f}±{std:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"operation_probe_latent_pos_{LATENT_POSITION}.png", dpi=150)
plt.show()
print(f"\nPlot saved to {OUTPUT_DIR}/operation_probe_latent_pos_{LATENT_POSITION}.png")

# %%
# Save results
results = {
    "config": {
        "checkpoint_path": CHECKPOINT_PATH,
        "base_model": BASE_MODEL,
        "num_latent": NUM_LATENT,
        "latent_position": LATENT_POSITION,
        "num_variations": NUM_VARIATIONS,
        "random_seed": RANDOM_SEED,
    },
    "training_template": TRAINING_TEMPLATE,
    "addition_results": {
        "overall_mean": addition_results["overall_mean"],
        "overall_std": addition_results["overall_std"],
        "per_template": addition_results["per_template"],
    },
    "subtraction_results": {
        "overall_mean": subtraction_results["overall_mean"],
        "overall_std": subtraction_results["overall_std"],
        "per_template": subtraction_results["per_template"],
    },
}

results_file = OUTPUT_DIR / f"operation_probe_results_latent_pos_{LATENT_POSITION}.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {results_file}")

# %%
# Print summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nLatent Position: {LATENT_POSITION}")
print(f"Number of variations per template: {NUM_VARIATIONS}")
print(f"\nProbe trained on: {TRAINING_TEMPLATE[:60]}...")
print(f"\nADDITION templates (n={len(ADDITION_TEMPLATES)}):")
print(f"  Mean dot product: {addition_results['overall_mean']:.4f}")
print(f"  Std dot product:  {addition_results['overall_std']:.4f}")
print(f"\nSUBTRACTION templates (n={len(SUBTRACTION_TEMPLATES)}):")
print(f"  Mean dot product: {subtraction_results['overall_mean']:.4f}")
print(f"  Std dot product:  {subtraction_results['overall_std']:.4f}")
print(
    f"\nDifference (Addition - Subtraction): {addition_results['overall_mean'] - subtraction_results['overall_mean']:.4f}"
)
print("=" * 80)
