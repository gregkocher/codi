# ABOUTME: Trains a linear direction probe on CODI latent vectors to classify operation type.
# ABOUTME: Evaluates and plots probe accuracy vs latent position across latent iterations.

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

# Output paths
OUTPUT_DIR = Path("results/operation_probe_latent_vectors")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Prompt templates

# Training templates (for computing the probe direction)
TRAINING_ADDITION_TEMPLATE = (
    "A team starts with {X} members. They recruit {Y} new members. "
    "Then each current member recruits {Z} additional people. "
    "How many people are there now on the team? Give the answer only and nothing else."
)
TRAINING_SUBTRACTION_TEMPLATE = (
    "A team starts with {X} members. {Y} members quit. "
    "Then each remaining member recruits {Z} additional people. "
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
def extract_latent_vectors(
    model,
    tokenizer,
    prompt: str,
    num_latent_iterations: int,
    sot_token: int,
) -> list[torch.Tensor]:
    """
    Extract CODI latent vectors for each latent position.

    The returned vectors correspond to the embeddings re-injected during latent
    iterations (i.e., post-projection if `model.use_prj` is enabled).

    Returns:
        List of length (num_latent_iterations + 1) where index i corresponds to
        latent position i, each tensor has shape [hidden_dim] on CPU.
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

    latents: list[torch.Tensor] = []

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

        # Latent position 0: latent embedding derived from the last token of final layer
        latent_raw = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        latent_embd = latent_raw
        if model.use_prj:
            latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)
        latents.append(latent_embd.squeeze(0).squeeze(0).cpu())

        # Latent iterations (positions 1..num_latent_iterations)
        for _ in range(num_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values

            latent_raw = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            latent_embd = latent_raw
            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)
            latents.append(latent_embd.squeeze(0).squeeze(0).cpu())

    return latents


# %%
def _l2_normalize(vec: torch.Tensor) -> torch.Tensor:
    denom = vec.norm()
    if denom.item() == 0:
        return vec
    return vec / denom


def train_probe_by_latent_position(
    pos_latents: dict[int, list[torch.Tensor]],
    neg_latents: dict[int, list[torch.Tensor]],
    num_latent_positions: int,
) -> dict[int, dict[str, torch.Tensor]]:
    """
    Train a simple linear direction probe separately for each latent position.

    For latent position p:
      - direction = normalize(mean(pos) - mean(neg))
      - threshold = midpoint between class means along that direction
    """
    probes: dict[int, dict[str, torch.Tensor]] = {}
    for latent_pos in range(num_latent_positions):
        pos_stack = torch.stack(pos_latents[latent_pos], dim=0).float()
        neg_stack = torch.stack(neg_latents[latent_pos], dim=0).float()

        pos_mean = pos_stack.mean(dim=0)
        neg_mean = neg_stack.mean(dim=0)
        direction = _l2_normalize(pos_mean - neg_mean)

        pos_proj_mean = torch.dot(direction, pos_mean).item()
        neg_proj_mean = torch.dot(direction, neg_mean).item()
        threshold = 0.5 * (pos_proj_mean + neg_proj_mean)

        probes[latent_pos] = {
            "direction": direction,
            "threshold": torch.tensor(threshold, dtype=torch.float32),
        }
    return probes


def evaluate_probe_accuracy_by_latent_position(
    probes: dict[int, dict[str, torch.Tensor]],
    pos_latents: dict[int, list[torch.Tensor]],
    neg_latents: dict[int, list[torch.Tensor]],
    num_latent_positions: int,
) -> dict[int, float]:
    accuracies: dict[int, float] = {}
    for latent_pos in range(num_latent_positions):
        direction = probes[latent_pos]["direction"].float()
        threshold = probes[latent_pos]["threshold"].item()

        correct = 0
        total = 0

        for vec in pos_latents[latent_pos]:
            score = torch.dot(direction, vec.float()).item()
            pred_pos = score > threshold
            correct += int(pred_pos)
            total += 1

        for vec in neg_latents[latent_pos]:
            score = torch.dot(direction, vec.float()).item()
            pred_pos = score > threshold
            correct += int(not pred_pos)
            total += 1

        accuracies[latent_pos] = correct / max(total, 1)
    return accuracies


def evaluate_probe_dot_products_by_latent_position(
    probes: dict[int, dict[str, torch.Tensor]],
    pos_latents: dict[int, list[torch.Tensor]],
    neg_latents: dict[int, list[torch.Tensor]],
    num_latent_positions: int,
) -> dict[int, dict[str, object]]:
    """
    Evaluate dot products (cosine similarity) with each latent-position probe direction.

    Returns per latent position:
      - addition: {mean, std, dot_products}
      - subtraction: {mean, std, dot_products}
    """
    results: dict[int, dict[str, object]] = {}
    for latent_pos in range(num_latent_positions):
        direction = probes[latent_pos]["direction"].float()

        add_dots: list[float] = []
        sub_dots: list[float] = []

        for vec in pos_latents[latent_pos]:
            vec_n = _l2_normalize(vec.float())
            add_dots.append(torch.dot(direction, vec_n).item())

        for vec in neg_latents[latent_pos]:
            vec_n = _l2_normalize(vec.float())
            sub_dots.append(torch.dot(direction, vec_n).item())

        results[latent_pos] = {
            "addition": {
                "mean": float(np.mean(add_dots)),
                "std": float(np.std(add_dots)),
                "dot_products": add_dots,
            },
            "subtraction": {
                "mean": float(np.mean(sub_dots)),
                "std": float(np.std(sub_dots)),
                "dot_products": sub_dots,
            },
        }

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

NUM_LATENT_POSITIONS = NUM_LATENT + 1

# %%
print("\nGenerating number variations...")
variations = generate_number_variations(NUM_VARIATIONS, RANDOM_SEED)
print(f"Generated {len(variations)} variations")
print(f"Sample variations: {variations[:5]}")

# %%
print("\nExtracting TRAINING latent vectors (addition vs subtraction)...")
train_pos_latents: dict[int, list[torch.Tensor]] = {
    i: [] for i in range(NUM_LATENT_POSITIONS)
}
train_neg_latents: dict[int, list[torch.Tensor]] = {
    i: [] for i in range(NUM_LATENT_POSITIONS)
}

for x, y, z in tqdm(variations, desc="Training prompts"):
    pos_prompt = TRAINING_ADDITION_TEMPLATE.format(X=x, Y=y, Z=z)
    neg_prompt = TRAINING_SUBTRACTION_TEMPLATE.format(X=x, Y=y, Z=z)

    pos_latents = extract_latent_vectors(
        model, tokenizer, pos_prompt, NUM_LATENT, SOT_TOKEN
    )
    neg_latents = extract_latent_vectors(
        model, tokenizer, neg_prompt, NUM_LATENT, SOT_TOKEN
    )

    for latent_pos in range(NUM_LATENT_POSITIONS):
        train_pos_latents[latent_pos].append(pos_latents[latent_pos])
        train_neg_latents[latent_pos].append(neg_latents[latent_pos])

probes = train_probe_by_latent_position(
    pos_latents=train_pos_latents,
    neg_latents=train_neg_latents,
    num_latent_positions=NUM_LATENT_POSITIONS,
)

# %%
print("\nExtracting TEST latent vectors (addition vs subtraction)...")
test_pos_latents: dict[int, list[torch.Tensor]] = {
    i: [] for i in range(NUM_LATENT_POSITIONS)
}
test_neg_latents: dict[int, list[torch.Tensor]] = {
    i: [] for i in range(NUM_LATENT_POSITIONS)
}

for template in tqdm(ADDITION_TEMPLATES, desc="Addition templates"):
    for x, y, z in variations:
        prompt = template.format(X=x, Y=y, Z=z)
        latents = extract_latent_vectors(
            model, tokenizer, prompt, NUM_LATENT, SOT_TOKEN
        )
        for latent_pos in range(NUM_LATENT_POSITIONS):
            test_pos_latents[latent_pos].append(latents[latent_pos])

for template in tqdm(SUBTRACTION_TEMPLATES, desc="Subtraction templates"):
    for x, y, z in variations:
        prompt = template.format(X=x, Y=y, Z=z)
        latents = extract_latent_vectors(
            model, tokenizer, prompt, NUM_LATENT, SOT_TOKEN
        )
        for latent_pos in range(NUM_LATENT_POSITIONS):
            test_neg_latents[latent_pos].append(latents[latent_pos])

accuracies = evaluate_probe_accuracy_by_latent_position(
    probes=probes,
    pos_latents=test_pos_latents,
    neg_latents=test_neg_latents,
    num_latent_positions=NUM_LATENT_POSITIONS,
)

# %%
dot_product_results = evaluate_probe_dot_products_by_latent_position(
    probes=probes,
    pos_latents=test_pos_latents,
    neg_latents=test_neg_latents,
    num_latent_positions=NUM_LATENT_POSITIONS,
)

# %%
print("\nDot products (cosine similarity) by latent position:")
for latent_pos in range(NUM_LATENT_POSITIONS):
    add_mean = dot_product_results[latent_pos]["addition"]["mean"]
    sub_mean = dot_product_results[latent_pos]["subtraction"]["mean"]
    print(
        f"  Latent Position {latent_pos}: addition={add_mean:.4f} subtraction={sub_mean:.4f}"
    )

# %%
# Save JSON results
results_file = OUTPUT_DIR / "probe_dot_products_by_latent_position.json"
with open(results_file, "w") as f:
    json.dump(
        {
            "config": {
                "checkpoint_path": CHECKPOINT_PATH,
                "base_model": BASE_MODEL,
                "num_latent": NUM_LATENT,
                "use_prj": USE_PRJ,
                "num_variations": NUM_VARIATIONS,
                "random_seed": RANDOM_SEED,
            },
            "dot_products": {str(k): v for k, v in dot_product_results.items()},
        },
        f,
        indent=2,
    )
print(f"\nSaved results to {results_file}")

# %%
# Plot dot products vs latent position
xs = list(range(NUM_LATENT_POSITIONS))
add_means = [dot_product_results[i]["addition"]["mean"] for i in xs]
add_stds = [dot_product_results[i]["addition"]["std"] for i in xs]
sub_means = [dot_product_results[i]["subtraction"]["mean"] for i in xs]
sub_stds = [dot_product_results[i]["subtraction"]["std"] for i in xs]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(xs, add_means, label="Addition", color="blue", linewidth=2, alpha=0.7)
ax.plot(xs, sub_means, label="Subtraction", color="red", linewidth=2, alpha=0.7)
ax.fill_between(
    xs,
    np.array(add_means) - np.array(add_stds),
    np.array(add_means) + np.array(add_stds),
    color="blue",
    alpha=0.2,
)
ax.fill_between(
    xs,
    np.array(sub_means) - np.array(sub_stds),
    np.array(sub_means) + np.array(sub_stds),
    color="red",
    alpha=0.2,
)
ax.set_xlabel("Latent Position", fontsize=12)
ax.set_ylabel("Dot Product with Probe Direction", fontsize=12)
ax.set_title(
    "Operation Probe Dot Products vs Latent Position (Latent Vectors)", fontsize=14
)
ax.grid(True, alpha=0.3)
ax.set_xticks(xs)
ax.legend()
plt.tight_layout()

plot_path = OUTPUT_DIR / "probe_dot_products_by_latent_position.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Saved plot to {plot_path}")
