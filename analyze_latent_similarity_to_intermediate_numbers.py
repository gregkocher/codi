# ABOUTME: Analyzes similarity between latent reasoning vectors and intermediate number token embeddings
# ABOUTME: Uses softmax-normalized probabilities over vocabulary for interpretable [0,1] similarity scores

# %%
"""
This script analyzes how latent reasoning vectors relate to number tokens.

Key methodology:
1. For each latent vector, compute dot product with ALL token embeddings in vocabulary
2. Apply softmax normalization to get a probability distribution (values sum to 1)
3. Extract probabilities for specific tokens of interest (intermediate numbers, final answer, etc.)
4. Average these probabilities across tokens and samples

This approach ensures:
- Similarities are in [0, 1] range and interpretable as probabilities
- Normalization accounts for the entire vocabulary context
- Results are comparable across different token sets
"""

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from src.model import CODI

# Load environment variables
load_dotenv()

# %%
# ==================== PARAMETERS ====================
# Model configuration
CHECKPOINT_PATH = "bcywinski/codi_llama1b-answer_only"
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
NUM_LATENT = 6
USE_PRJ = True

# Inference configuration
NUM_LATENT_ITERATIONS = 6
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.1
GREEDY = True

# Data configuration
INTERMEDIATE_RESULTS_FILE = "multi_arith_intermediate_results.json"
MAX_SAMPLES = None  # Set to None to process all samples, or a number to limit

# Baseline configuration
NUM_RANDOM_NUMBERS = 100  # How many random numbers to use for baseline
RANDOM_NUMBER_RANGE = (0, 1000)  # Range for random numbers

# Output configuration
OUTPUT_PLOT = "latent_similarity_analysis.png"
OUTPUT_DATA = "latent_similarity_data.json"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# ==================== LOAD MODEL ====================
print("Loading CODI model...")
model = CODI.from_pretrained(
    checkpoint_path=CHECKPOINT_PATH,
    model_name_or_path=BASE_MODEL,
    num_latent=NUM_LATENT,
    use_prj=USE_PRJ,
    device=DEVICE,
)

tokenizer = model.tokenizer
tokenizer.add_special_tokens({"additional_special_tokens": ["<|bocot|>", "<|eocot|>"]})

# Get special token IDs
sot_token = tokenizer.convert_tokens_to_ids("<|bocot|>")
eot_token = tokenizer.convert_tokens_to_ids("<|eocot|>")

print(f"Model loaded on {DEVICE}")

# %%
# ==================== LOAD DATA ====================
print(f"Loading intermediate results from {INTERMEDIATE_RESULTS_FILE}...")
with open(INTERMEDIATE_RESULTS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

if MAX_SAMPLES is not None:
    data = data[:MAX_SAMPLES]
    print(f"Limited to {MAX_SAMPLES} samples")

print(f"Loaded {len(data)} samples")

# %%
# ==================== GET TOKEN EMBEDDINGS ====================
print("Extracting token embedding matrix...")
embed_matrix = model.codi.model.model.embed_tokens.weight.to(
    "cpu"
)  # [vocab_size, hidden_dim]
vocab_size, hidden_dim = embed_matrix.shape
print(f"Embedding matrix shape: {embed_matrix.shape}")

# %%
# ==================== HELPER FUNCTIONS ====================


def extract_numbers_from_text(text):
    """Extract all numbers from text.

    Args:
        text: String containing numbers

    Returns:
        List of numbers (as strings)
    """
    import re

    # Match integers and decimals
    numbers = re.findall(r"\b\d+\.?\d*\b", text)
    return numbers


def get_number_token_ids(numbers, tokenizer):
    """Get token IDs for a list of numbers.

    Args:
        numbers: List of numbers (can be strings or ints)
        tokenizer: The tokenizer

    Returns:
        List of token IDs
    """
    token_ids = []
    for num in numbers:
        # Tokenize the number - it might be split into multiple tokens
        ids = tokenizer.encode(str(num), add_special_tokens=False)
        token_ids.extend(ids)

    return token_ids


def compute_softmax_similarities(latent_vectors, embed_matrix):
    """Compute softmax-normalized similarities to all tokens in vocabulary.

    For each latent vector:
    1. Compute dot product with all token embeddings
    2. Apply softmax to get probability distribution over vocabulary
    3. Return the normalized similarities

    Args:
        latent_vectors: [num_latents, hidden_dim]
        embed_matrix: [vocab_size, hidden_dim]

    Returns:
        similarities: [num_latents, vocab_size] - softmax normalized, values in [0, 1]
    """
    # Compute similarities to all tokens
    similarities = latent_vectors @ embed_matrix.T  # [num_latents, vocab_size]

    # Apply softmax to normalize to probability distribution
    similarities = torch.nn.functional.softmax(similarities, dim=-1)

    return similarities


def extract_token_similarities(softmax_sims, token_ids):
    """Extract similarities for specific token IDs from softmax-normalized similarities.

    Args:
        softmax_sims: [num_latents, vocab_size] - softmax normalized similarities
        token_ids: List of token IDs to extract

    Returns:
        extracted_sims: [num_latents, num_tokens] - similarities for specified tokens
    """
    if len(token_ids) == 0:
        return None

    # Filter out invalid token IDs
    valid_token_ids = [tid for tid in token_ids if tid < softmax_sims.shape[1]]

    if len(valid_token_ids) == 0:
        return None

    # Extract similarities for specific tokens
    extracted_sims = softmax_sims[:, valid_token_ids]  # [num_latents, num_tokens]

    return extracted_sims


# %%
# ==================== PROCESS EACH SAMPLE ====================
print("\nProcessing samples and computing similarities...")

# Store results for each latent position
latent_position_similarities = [[] for _ in range(NUM_LATENT_ITERATIONS + 1)]
latent_position_baseline_random = [[] for _ in range(NUM_LATENT_ITERATIONS + 1)]
latent_position_baseline_all = [[] for _ in range(NUM_LATENT_ITERATIONS + 1)]
latent_position_baseline_final_answer = [[] for _ in range(NUM_LATENT_ITERATIONS + 1)]
latent_position_baseline_prompt_numbers = [[] for _ in range(NUM_LATENT_ITERATIONS + 1)]

# Generate random numbers for baseline once
print(f"Generating {NUM_RANDOM_NUMBERS} random numbers for baseline...")
random_numbers = [
    np.random.randint(RANDOM_NUMBER_RANGE[0], RANDOM_NUMBER_RANGE[1])
    for _ in range(NUM_RANDOM_NUMBERS)
]
random_number_token_ids = get_number_token_ids(random_numbers, tokenizer)
print(f"Random numbers generated: {len(random_number_token_ids)} tokens")

samples_processed = 0
samples_with_intermediate = 0

for idx, sample in enumerate(tqdm(data, desc="Processing samples")):
    question = sample["question"]
    intermediate_numbers = sample.get("intermediate_numbers", [])
    final_answer = sample.get("final_answer", "")

    # Skip samples without intermediate numbers
    if not intermediate_numbers or len(intermediate_numbers) == 0:
        continue

    samples_with_intermediate += 1

    # Tokenize and run inference
    inputs = tokenizer(question, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    try:
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            num_latent_iterations=NUM_LATENT_ITERATIONS,
            temperature=TEMPERATURE,
            greedy=GREEDY,
            return_latent_vectors=True,
            sot_token=sot_token,
            eot_token=eot_token,
            remove_eos=False,
            output_hidden_states=True,
        )

        # Extract latent vectors
        latent_vectors = (
            torch.stack(output["latent_vectors"]).squeeze(1).squeeze(1).to("cpu")
        )
        # Shape: [num_latent_iterations + 1, hidden_dim]

        # Compute softmax-normalized similarities to ALL tokens in vocabulary
        # This gives us a probability distribution over the vocabulary for each latent vector
        softmax_sims = compute_softmax_similarities(latent_vectors, embed_matrix)
        # Shape: [num_latents, vocab_size]

        # Get token IDs for intermediate numbers
        intermediate_token_ids = get_number_token_ids(intermediate_numbers, tokenizer)

        if len(intermediate_token_ids) == 0:
            continue

        # Extract similarities for intermediate number tokens
        intermediate_sims = extract_token_similarities(
            softmax_sims, intermediate_token_ids
        )
        # Shape: [num_latents, num_intermediate_tokens]

        # Average across all intermediate number tokens for each latent position
        avg_similarities = intermediate_sims.mean(dim=-1)  # [num_latents]

        for latent_idx in range(len(avg_similarities)):
            latent_position_similarities[latent_idx].append(
                avg_similarities[latent_idx].item()
            )

        # Extract similarities for final answer tokens
        final_answer_token_ids = get_number_token_ids([final_answer], tokenizer)

        if len(final_answer_token_ids) > 0:
            final_answer_sims = extract_token_similarities(
                softmax_sims, final_answer_token_ids
            )
            avg_final_answer_similarities = final_answer_sims.mean(
                dim=-1
            )  # [num_latents]

            for latent_idx in range(len(avg_final_answer_similarities)):
                latent_position_baseline_final_answer[latent_idx].append(
                    avg_final_answer_similarities[latent_idx].item()
                )

        # Extract similarities for prompt numbers
        prompt_numbers = extract_numbers_from_text(question)
        prompt_number_token_ids = get_number_token_ids(prompt_numbers, tokenizer)

        if len(prompt_number_token_ids) > 0:
            prompt_number_sims = extract_token_similarities(
                softmax_sims, prompt_number_token_ids
            )
            avg_prompt_number_similarities = prompt_number_sims.mean(
                dim=-1
            )  # [num_latents]

            for latent_idx in range(len(avg_prompt_number_similarities)):
                latent_position_baseline_prompt_numbers[latent_idx].append(
                    avg_prompt_number_similarities[latent_idx].item()
                )

        # Baseline 1: Similarity to random numbers
        if len(random_number_token_ids) > 0:
            random_sims = extract_token_similarities(
                softmax_sims, random_number_token_ids
            )
            avg_random_similarities = random_sims.mean(dim=-1)  # [num_latents]

            for latent_idx in range(len(avg_random_similarities)):
                latent_position_baseline_random[latent_idx].append(
                    avg_random_similarities[latent_idx].item()
                )

        # Baseline 2: Average similarity across all tokens in vocabulary
        # Since softmax_sims is a probability distribution that sums to 1, the mean
        # equals 1/vocab_size (uniform distribution baseline)
        # This represents the expected probability for a random token
        avg_all_similarities = softmax_sims.mean(dim=-1)  # [num_latents]

        for latent_idx in range(len(avg_all_similarities)):
            latent_position_baseline_all[latent_idx].append(
                avg_all_similarities[latent_idx].item()
            )

        samples_processed += 1

    except Exception as e:
        print(f"\nError processing sample {idx}: {e}")
        continue

print(f"\nProcessed {samples_processed} samples")
print(f"Samples with intermediate numbers: {samples_with_intermediate}")

# %%
# ==================== COMPUTE AVERAGES ====================
print("\nComputing average similarities across all samples...")

# Average and std dev across all samples for each latent position
avg_similarity_per_latent = []
std_similarity_per_latent = []
avg_random_per_latent = []
std_random_per_latent = []
avg_all_per_latent = []
std_all_per_latent = []
avg_final_answer_per_latent = []
std_final_answer_per_latent = []
avg_prompt_numbers_per_latent = []
std_prompt_numbers_per_latent = []

for latent_idx in range(NUM_LATENT_ITERATIONS + 1):
    if len(latent_position_similarities[latent_idx]) > 0:
        avg_similarity_per_latent.append(
            np.mean(latent_position_similarities[latent_idx])
        )
        std_similarity_per_latent.append(
            np.std(latent_position_similarities[latent_idx])
        )
        avg_random_per_latent.append(
            np.mean(latent_position_baseline_random[latent_idx])
        )
        std_random_per_latent.append(
            np.std(latent_position_baseline_random[latent_idx])
        )
        avg_all_per_latent.append(np.mean(latent_position_baseline_all[latent_idx]))
        std_all_per_latent.append(np.std(latent_position_baseline_all[latent_idx]))
        avg_final_answer_per_latent.append(
            np.mean(latent_position_baseline_final_answer[latent_idx])
        )
        std_final_answer_per_latent.append(
            np.std(latent_position_baseline_final_answer[latent_idx])
        )
        avg_prompt_numbers_per_latent.append(
            np.mean(latent_position_baseline_prompt_numbers[latent_idx])
        )
        std_prompt_numbers_per_latent.append(
            np.std(latent_position_baseline_prompt_numbers[latent_idx])
        )
    else:
        avg_similarity_per_latent.append(0)
        std_similarity_per_latent.append(0)
        avg_random_per_latent.append(0)
        std_random_per_latent.append(0)
        avg_all_per_latent.append(0)
        std_all_per_latent.append(0)
        avg_final_answer_per_latent.append(0)
        std_final_answer_per_latent.append(0)
        avg_prompt_numbers_per_latent.append(0)
        std_prompt_numbers_per_latent.append(0)

print("Average similarities and standard deviations computed")

# %%
# ==================== SAVE RESULTS ====================
print(f"\nSaving results to {OUTPUT_DATA}...")

results = {
    "num_samples": samples_processed,
    "num_latent_iterations": NUM_LATENT_ITERATIONS,
    "avg_similarity_to_intermediate": avg_similarity_per_latent,
    "std_similarity_to_intermediate": std_similarity_per_latent,
    "avg_similarity_to_final_answer": avg_final_answer_per_latent,
    "std_similarity_to_final_answer": std_final_answer_per_latent,
    "avg_similarity_to_prompt_numbers": avg_prompt_numbers_per_latent,
    "std_similarity_to_prompt_numbers": std_prompt_numbers_per_latent,
    "avg_similarity_to_random_numbers": avg_random_per_latent,
    "std_similarity_to_random_numbers": std_random_per_latent,
    "avg_similarity_to_all_tokens": avg_all_per_latent,
    "std_similarity_to_all_tokens": std_all_per_latent,
    "raw_data": {
        "intermediate_similarities": [
            latent_position_similarities[i]
            for i in range(len(latent_position_similarities))
        ],
        "final_answer_similarities": [
            latent_position_baseline_final_answer[i]
            for i in range(len(latent_position_baseline_final_answer))
        ],
        "prompt_number_similarities": [
            latent_position_baseline_prompt_numbers[i]
            for i in range(len(latent_position_baseline_prompt_numbers))
        ],
        "random_similarities": [
            latent_position_baseline_random[i]
            for i in range(len(latent_position_baseline_random))
        ],
        "all_token_similarities": [
            latent_position_baseline_all[i]
            for i in range(len(latent_position_baseline_all))
        ],
    },
}

with open(OUTPUT_DATA, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {OUTPUT_DATA}")

# %%
# ==================== PLOT RESULTS ====================
print("\nCreating plot...")

plt.figure(figsize=(12, 7))

x_positions = np.array(range(NUM_LATENT_ITERATIONS + 1))

# Convert to numpy arrays for easier computation
avg_similarity_per_latent_np = np.array(avg_similarity_per_latent)
std_similarity_per_latent_np = np.array(std_similarity_per_latent)
avg_final_answer_per_latent_np = np.array(avg_final_answer_per_latent)
std_final_answer_per_latent_np = np.array(std_final_answer_per_latent)
avg_prompt_numbers_per_latent_np = np.array(avg_prompt_numbers_per_latent)
std_prompt_numbers_per_latent_np = np.array(std_prompt_numbers_per_latent)
avg_random_per_latent_np = np.array(avg_random_per_latent)
std_random_per_latent_np = np.array(std_random_per_latent)
avg_all_per_latent_np = np.array(avg_all_per_latent)
std_all_per_latent_np = np.array(std_all_per_latent)

# Plot main line: similarity to intermediate numbers
plt.plot(
    x_positions,
    avg_similarity_per_latent_np,
    marker="o",
    linewidth=3,
    markersize=10,
    label="Intermediate Numbers",
    color="blue",
    zorder=3,
)
plt.fill_between(
    x_positions,
    avg_similarity_per_latent_np - std_similarity_per_latent_np,
    avg_similarity_per_latent_np + std_similarity_per_latent_np,
    color="blue",
    alpha=0.2,
    zorder=1,
)

# Plot baseline: final answer
plt.plot(
    x_positions,
    avg_final_answer_per_latent_np,
    marker="D",
    linewidth=3,
    markersize=10,
    label="Final Answer",
    color="green",
    linestyle="-",
    zorder=3,
)
plt.fill_between(
    x_positions,
    avg_final_answer_per_latent_np - std_final_answer_per_latent_np,
    avg_final_answer_per_latent_np + std_final_answer_per_latent_np,
    color="green",
    alpha=0.2,
    zorder=1,
)

# Plot baseline: prompt numbers
plt.plot(
    x_positions,
    avg_prompt_numbers_per_latent_np,
    marker="s",
    linewidth=3,
    markersize=10,
    label="Prompt Numbers",
    color="purple",
    linestyle="-.",
    zorder=3,
)
plt.fill_between(
    x_positions,
    avg_prompt_numbers_per_latent_np - std_prompt_numbers_per_latent_np,
    avg_prompt_numbers_per_latent_np + std_prompt_numbers_per_latent_np,
    color="purple",
    alpha=0.2,
    zorder=1,
)

# Plot baseline: random numbers
plt.plot(
    x_positions,
    avg_random_per_latent_np,
    marker="s",
    linewidth=3,
    markersize=10,
    label="Random Numbers",
    color="red",
    linestyle="--",
    zorder=3,
)
plt.fill_between(
    x_positions,
    avg_random_per_latent_np - std_random_per_latent_np,
    avg_random_per_latent_np + std_random_per_latent_np,
    color="red",
    alpha=0.2,
    zorder=1,
)

# Plot baseline: all tokens
plt.plot(
    x_positions,
    avg_all_per_latent_np,
    marker="^",
    linewidth=3,
    markersize=10,
    label="All Token Embeddings",
    color="gray",
    linestyle=":",
    zorder=3,
)
plt.fill_between(
    x_positions,
    avg_all_per_latent_np - std_all_per_latent_np,
    avg_all_per_latent_np + std_all_per_latent_np,
    color="gray",
    alpha=0.15,
    zorder=1,
)

plt.xlabel("Latent Vector Index", fontsize=16)
plt.ylabel("Average Similarity", fontsize=16)
plt.title(
    "Latent Vector Similarity to Token Embeddings",
    fontsize=22,
)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.2)
plt.xticks(x_positions, fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
print(f"Plot saved to {OUTPUT_PLOT}")

plt.show()

# %%
# ==================== PRINT SUMMARY ====================
print("\n" + "=" * 130)
print("SUMMARY")
print("=" * 130)
print(f"Samples processed: {samples_processed}")
print(f"Latent iterations: {NUM_LATENT_ITERATIONS}")
print()
print("Average Softmax Probabilities (Mean ± Std Dev, values in [0, 1]):")
print("-" * 130)
print(
    f"{'Latent':<8} {'Intermediate':<20} {'Final Ans':<20} {'Prompt Nums':<20} "
    f"{'Random':<20} {'All Tokens':<20}"
)
print("-" * 130)
for i in range(NUM_LATENT_ITERATIONS + 1):
    print(
        f"{i:<8} "
        f"{avg_similarity_per_latent[i]:.4f}±{std_similarity_per_latent[i]:.4f}    "
        f"{avg_final_answer_per_latent[i]:.4f}±{std_final_answer_per_latent[i]:.4f}    "
        f"{avg_prompt_numbers_per_latent[i]:.4f}±{std_prompt_numbers_per_latent[i]:.4f}    "
        f"{avg_random_per_latent[i]:.4f}±{std_random_per_latent[i]:.4f}    "
        f"{avg_all_per_latent[i]:.4f}±{std_all_per_latent[i]:.4f}"
    )
print("-" * 130)

# Compute differences
print("\nDifference: Intermediate Numbers vs Final Answer:")
for i in range(NUM_LATENT_ITERATIONS + 1):
    diff = avg_similarity_per_latent[i] - avg_final_answer_per_latent[i]
    print(f"  Latent {i}: {diff:+.6f}")

print("\nDifference: Intermediate Numbers vs Prompt Numbers:")
for i in range(NUM_LATENT_ITERATIONS + 1):
    diff = avg_similarity_per_latent[i] - avg_prompt_numbers_per_latent[i]
    print(f"  Latent {i}: {diff:+.6f}")

print("\nDifference: Intermediate Numbers vs Random Numbers:")
for i in range(NUM_LATENT_ITERATIONS + 1):
    diff = avg_similarity_per_latent[i] - avg_random_per_latent[i]
    print(f"  Latent {i}: {diff:+.6f}")

print("\nDifference: Intermediate Numbers vs All Tokens:")
for i in range(NUM_LATENT_ITERATIONS + 1):
    diff = avg_similarity_per_latent[i] - avg_all_per_latent[i]
    print(f"  Latent {i}: {diff:+.6f}")

print("\n" + "=" * 130)
print("Analysis complete!")
print("=" * 130)
