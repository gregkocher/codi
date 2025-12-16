# ABOUTME: Analyzes latent vector similarity to intermediate computation results across multiple prompt variations
# ABOUTME: Computes average similarity differences between target tokens and baseline, then plots averaged results

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from src.model import CODI

load_dotenv()

# %%
# Load model
model = CODI.from_pretrained(
    checkpoint_path="bcywinski/codi_llama1b-answer_only",
    model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
    lora_r=128,
    lora_alpha=32,
    num_latent=6,
    use_prj=True,
    device="cuda",
    dtype="bfloat16",
    strict=False,
    checkpoint_save_path="./checkpoints/bcywinski/codi_llama1b-answer_only",
    remove_eos=True,
    full_precision=True,
)

# %%
# Setup tokenizer
tokenizer = model.tokenizer

if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token_id = model.pad_token_id
tokenizer.add_special_tokens({"additional_special_tokens": ["<|bocot|>", "<|eocot|>"]})
tokenizer.bocot_id = tokenizer.convert_tokens_to_ids("<|bocot|>")
tokenizer.eocot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")

# %%
# Define 10 different X, Y, Z combinations
xyz_combinations = [
    (3, 2, 4),  # intermediate_1: 5, intermediate_2: 20, final: 25
    (5, 3, 2),  # intermediate_1: 8, intermediate_2: 16, final: 24
    (2, 4, 3),  # intermediate_1: 6, intermediate_2: 18, final: 24
    (4, 1, 5),  # intermediate_1: 5, intermediate_2: 25, final: 30
    (1, 3, 4),  # intermediate_1: 4, intermediate_2: 16, final: 20
    (6, 2, 3),  # intermediate_1: 8, intermediate_2: 24, final: 32
    (3, 5, 2),  # intermediate_1: 8, intermediate_2: 16, final: 24
    (2, 2, 5),  # intermediate_1: 4, intermediate_2: 20, final: 24
    (5, 1, 3),  # intermediate_1: 6, intermediate_2: 18, final: 24
    (4, 4, 2),  # intermediate_1: 8, intermediate_2: 16, final: 24
]

# %%
# Storage for all similarities
# Will store: [num_variations, 3_targets, num_layers, num_tokens]
all_differences = []

# %%
# Generation parameters
skip_thinking = False
verbalize_cot = False
max_new_tokens = 256
num_latent_iterations = 6
temperature = 0.1
top_k = 40
top_p = 0.95

# %%
# Process each X, Y, Z combination
for idx, (X, Y, Z) in enumerate(xyz_combinations):
    print(f"\n{'=' * 60}")
    print(f"Processing combination {idx + 1}/10: X={X}, Y={Y}, Z={Z}")
    print(f"{'=' * 60}")

    # Calculate intermediate results and final answer
    intermediate_1 = X + Y  # After recruiting Y new members
    intermediate_2 = (X + Y) * Z  # Number of newly recruited people
    final_answer = (X + Y) + ((X + Y) * Z)  # Total people on team

    print(f"Intermediate 1: {intermediate_1}")
    print(f"Intermediate 2: {intermediate_2}")
    print(f"Final answer: {final_answer}")

    # Create prompt
    prompt = f"""A team starts with {X} members. They recruit {Y} new members. Then each current member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else."""

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(model.codi.device)
    attention_mask = inputs["attention_mask"].to(model.codi.device)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        num_latent_iterations=num_latent_iterations,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        greedy=True,
        return_latent_vectors=True,
        remove_eos=False,
        output_attentions=False,
        skip_thinking=skip_thinking,
        verbalize_cot=verbalize_cot,
        output_hidden_states=True,
        sot_token=tokenizer.convert_tokens_to_ids("<|bocot|>"),
        eot_token=tokenizer.convert_tokens_to_ids("<|eocot|>"),
    )

    # output["hidden_states"] is (seq_len, n_layers, hidden_dim) for batch size 1.
    hs = output["hidden_states"].to("cpu")
    hs_squeezed = hs.permute(1, 0, 2)

    # Normalize hidden states
    hs_norm = hs_squeezed / hs_squeezed.norm(dim=-1, keepdim=True)
    hs_norm = hs_norm.float()

    # Get embedding matrix
    embedding_matrix = model.codi.base_model.model.model.embed_tokens.weight

    # Convert intermediate results and final answer to token strings
    target_tokens = [str(intermediate_1), str(intermediate_2), str(final_answer)]

    # Storage for this variation's differences
    variation_differences = []

    # For each intermediate result
    for target_idx, target_token_str in enumerate(target_tokens):
        print(
            f"\nAnalyzing similarity to intermediate {target_idx + 1}: '{target_token_str}'"
        )

        # Get token id
        target_token_id = tokenizer.convert_tokens_to_ids(target_token_str)
        if (
            target_token_id is None
            or target_token_id < 0
            or target_token_id >= embedding_matrix.shape[0]
        ):
            print(f"Warning: Token '{target_token_str}' not found, skipping")
            continue

        # Get target token embedding and normalize
        token_embedding = embedding_matrix[target_token_id].cpu().float()
        token_embedding_norm = token_embedding / token_embedding.norm()

        # Compute similarity to target token
        target_cosine_sim = (
            torch.einsum("ltd,d->lt", hs_norm, token_embedding_norm)
            .cpu()
            .float()
            .numpy()
        )

        # Compute baseline: average similarity to 1000 number tokens
        np.random.seed(42)
        valid_number_token_ids = []
        num_to_find = 1000
        max_num_checked = 10000
        checked = 0
        i = 0
        vocab_size = embedding_matrix.shape[0]

        while len(valid_number_token_ids) < num_to_find and checked < max_num_checked:
            token_str = str(i)
            tid = tokenizer.convert_tokens_to_ids(token_str)
            if tid is not None and 0 <= tid < vocab_size:
                valid_number_token_ids.append(tid)
            i += 1
            checked += 1

        baseline_token_ids = valid_number_token_ids[:num_to_find]
        baseline_embeds = embedding_matrix[baseline_token_ids, :].cpu().float()
        baseline_embeds_norm = baseline_embeds / baseline_embeds.norm(
            dim=1, keepdim=True
        )

        # Compute baseline similarities
        cosines_baseline = (
            torch.einsum("ltd,bd->ltb", hs_norm, baseline_embeds_norm)
            .cpu()
            .float()
            .numpy()
        )
        baseline_avg = cosines_baseline.mean(axis=-1)

        # Calculate difference: target - baseline
        difference = target_cosine_sim - baseline_avg

        variation_differences.append(difference)

    all_differences.append(variation_differences)

# %%
# Convert to numpy array: [num_variations, 3_targets, num_layers, num_tokens]
# Note: num_tokens might vary, so we'll need to handle this carefully
# For now, find the minimum number of tokens across all variations
min_tokens = min([diff.shape[1] for var_diffs in all_differences for diff in var_diffs])

# Crop all to minimum token length and stack
all_differences_array = np.array(
    [[diff[:, :min_tokens] for diff in var_diffs] for var_diffs in all_differences]
)

print(f"\nAll differences shape: {all_differences_array.shape}")
print(
    f"Shape: [num_variations={all_differences_array.shape[0]}, num_targets={all_differences_array.shape[1]}, num_layers={all_differences_array.shape[2]}, num_tokens={all_differences_array.shape[3]}]"
)

# %%
# Average across variations: [3_targets, num_layers, num_tokens]
avg_differences = all_differences_array.mean(axis=0)

print(f"\nAveraged differences shape: {avg_differences.shape}")
print(
    f"Shape: [num_targets={avg_differences.shape[0]}, num_layers={avg_differences.shape[1]}, num_tokens={avg_differences.shape[2]}]"
)

# %%
# Get token strings for x-axis (using first variation's prompt)
X, Y, Z = xyz_combinations[0]
prompt = f"""A team starts with {X} members. They recruit {Y} new members. Then each current member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else."""
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
token_ids_first = inputs["input_ids"][0, :min_tokens].tolist()
token_strs = tokenizer.convert_ids_to_tokens(token_ids_first)

# %%
# Create plot with 3 subplots (two intermediate results + final answer)
fig, axes = plt.subplots(3, 1, figsize=(14, 16))

target_labels = [
    "Intermediate Result 1 (X+Y)",
    "Intermediate Result 2 ((X+Y)*Z)",
    "Final Answer ((X+Y)+((X+Y)*Z))",
]

for target_idx in range(3):
    ax = axes[target_idx]

    # Get the averaged difference for this target
    diff_data = avg_differences[target_idx]  # [num_layers, num_tokens]

    # Plot heatmap
    im = ax.imshow(
        diff_data,
        aspect="auto",
        cmap="bwr",
        vmin=-np.max(np.abs(diff_data)),
        vmax=np.max(np.abs(diff_data)),
    )

    plt.colorbar(im, ax=ax, label="Avg similarity difference")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Token Position")
    ax.set_xticks(np.arange(len(token_strs)))
    ax.set_xticklabels(token_strs, rotation=90, fontsize=8)
    ax.set_title(f"{target_labels[target_idx]} (averaged over 10 variations)")

plt.tight_layout()
plt.savefig(
    "averaged_intermediate_results_similarity.png", dpi=150, bbox_inches="tight"
)
plt.show()

print("\nPlot saved as 'averaged_intermediate_results_similarity.png'")

# %%
# Print statistics
target_names = ["Intermediate Result 1", "Intermediate Result 2", "Final Answer"]

for target_idx in range(3):
    print(f"\n{'=' * 60}")
    print(f"Statistics for {target_names[target_idx]}")
    print(f"{'=' * 60}")

    diff_data = avg_differences[target_idx]

    print(f"Max difference: {diff_data.max():.6f}")
    print(f"Min difference: {diff_data.min():.6f}")
    print(f"Mean absolute difference: {np.abs(diff_data).mean():.6f}")

    # Find layer and token with maximum difference
    max_layer, max_token = np.unravel_index(diff_data.argmax(), diff_data.shape)
    print(
        f"Max at layer {max_layer}, token position {max_token} ('{token_strs[max_token]}'): {diff_data[max_layer, max_token]:.6f}"
    )

# %%
