# ABOUTME: Analyzes similarity between latent reasoning vectors and mathematical operation token embeddings
# ABOUTME: Compares correct operations vs incorrect operations using softmax-normalized probabilities

# %%
"""
This script analyzes how latent reasoning vectors relate to mathematical operation tokens.

Key methodology:
1. For each latent vector, compute dot product with ALL token embeddings in vocabulary
2. Apply softmax normalization to get a probability distribution (values sum to 1)
3. Extract probabilities for operation tokens: +, -, *, /
4. Compare similarity to correct operations vs incorrect operations

This approach helps understand whether latent vectors encode the mathematical operations
needed for intermediate calculations.
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
OPERATIONS_FILE = "multi_arith_intermediate_operations.json"
MAX_SAMPLES = None  # Set to None to process all samples, or a number to limit

# Baseline configuration
NUM_RANDOM_TOKENS = 1000  # Random tokens for baseline

# Mathematical operations to analyze
OPERATIONS = ["+", "-", "*", "/"]

# Verbalized operations mapping
OPERATION_WORDS = {
    "+": ["add", "plus", "addition"],
    "-": ["subtract", "minus", "subtraction"],
    "*": ["multiply", "times", "multiplication"],
    "/": ["divide", "division"],
}

# Output configuration
OUTPUT_PLOT = "latent_similarity_to_operations.png"
OUTPUT_DATA = "latent_similarity_to_operations_data.json"

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
print(f"Loading operations from {OPERATIONS_FILE}...")
with open(OPERATIONS_FILE, encoding="utf-8") as f:
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

# Get token IDs for mathematical operations (symbolic)
print("\nMathematical operation token IDs (symbolic):")
operation_token_ids = {}
for op in OPERATIONS:
    # Try different tokenization approaches
    token_ids = tokenizer.encode(op, add_special_tokens=False)
    if len(token_ids) > 0:
        operation_token_ids[op] = token_ids[0]  # Take first token
        print(f"  '{op}' -> {token_ids[0]} ('{tokenizer.decode([token_ids[0]])}')")
    else:
        print(f"  '{op}' -> WARNING: Could not tokenize!")

# Get token IDs for verbalized operations
print("\nVerbalized operation token IDs:")
operation_word_token_ids = {}
for op, words in OPERATION_WORDS.items():
    operation_word_token_ids[op] = []
    for word in words:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(token_ids) > 0:
            operation_word_token_ids[op].append(token_ids[0])  # Take first token
            print(
                f"  '{op}' -> '{word}' -> {token_ids[0]} ('{tokenizer.decode([token_ids[0]])}')"
            )
        else:
            print(f"  '{op}' -> '{word}' -> WARNING: Could not tokenize!")

# %%
# ==================== HELPER FUNCTIONS ====================


def compute_softmax_similarities(latent_vectors, embed_matrix):
    """Compute softmax-normalized similarities to all tokens in vocabulary.

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
    """Extract similarities for specific token IDs.

    Args:
        softmax_sims: [num_latents, vocab_size]
        token_ids: List of token IDs

    Returns:
        extracted_sims: [num_latents, num_tokens]
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
latent_position_correct_ops = [[] for _ in range(NUM_LATENT_ITERATIONS + 1)]
latent_position_correct_ops_verbal = [[] for _ in range(NUM_LATENT_ITERATIONS + 1)]
latent_position_incorrect_ops = [[] for _ in range(NUM_LATENT_ITERATIONS + 1)]
latent_position_incorrect_ops_verbal = [[] for _ in range(NUM_LATENT_ITERATIONS + 1)]
latent_position_baseline_random = [[] for _ in range(NUM_LATENT_ITERATIONS + 1)]

# Generate random token IDs for baseline
print(f"Generating {NUM_RANDOM_TOKENS} random token IDs for baseline...")
random_token_ids = np.random.randint(0, vocab_size, NUM_RANDOM_TOKENS).tolist()
print(f"Random tokens generated: {len(random_token_ids)} tokens")

samples_processed = 0
samples_with_operations = 0

for sample in tqdm(data, desc="Processing samples"):
    idx = sample["index"]
    question = sample["question"]
    intermediate_operations = sample.get("intermediate_operations", [])

    # Skip samples without intermediate operations
    if not intermediate_operations or len(intermediate_operations) == 0:
        continue

    samples_with_operations += 1

    # Get correct and incorrect operation token IDs (symbolic)
    correct_op_token_ids = []
    incorrect_op_token_ids = []

    for op in intermediate_operations:
        if op in operation_token_ids:
            correct_op_token_ids.append(operation_token_ids[op])

    # Get incorrect operations (all operations not used)
    for op in OPERATIONS:
        if op not in intermediate_operations and op in operation_token_ids:
            incorrect_op_token_ids.append(operation_token_ids[op])

    # Get correct and incorrect verbal operation token IDs
    correct_op_verbal_token_ids = []
    incorrect_op_verbal_token_ids = []

    for op in intermediate_operations:
        if op in operation_word_token_ids:
            correct_op_verbal_token_ids.extend(operation_word_token_ids[op])

    # Get incorrect verbal operations (all operations not used)
    for op in OPERATIONS:
        if op not in intermediate_operations and op in operation_word_token_ids:
            incorrect_op_verbal_token_ids.extend(operation_word_token_ids[op])

    if len(correct_op_token_ids) == 0:
        continue

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

        # Compute softmax-normalized similarities to ALL tokens
        softmax_sims = compute_softmax_similarities(latent_vectors, embed_matrix)
        # Shape: [num_latents, vocab_size]

        # Extract similarities for correct operations (symbolic)
        correct_op_sims = extract_token_similarities(softmax_sims, correct_op_token_ids)
        # Shape: [num_latents, num_correct_ops]

        # Average across all correct operation tokens for each latent position
        avg_correct_sims = correct_op_sims.mean(dim=-1)  # [num_latents]

        for latent_idx in range(len(avg_correct_sims)):
            latent_position_correct_ops[latent_idx].append(
                avg_correct_sims[latent_idx].item()
            )

        # Extract similarities for correct operations (verbal)
        if len(correct_op_verbal_token_ids) > 0:
            correct_op_verbal_sims = extract_token_similarities(
                softmax_sims, correct_op_verbal_token_ids
            )
            avg_correct_verbal_sims = correct_op_verbal_sims.mean(
                dim=-1
            )  # [num_latents]

            for latent_idx in range(len(avg_correct_verbal_sims)):
                latent_position_correct_ops_verbal[latent_idx].append(
                    avg_correct_verbal_sims[latent_idx].item()
                )

        # Extract similarities for incorrect operations (symbolic)
        if len(incorrect_op_token_ids) > 0:
            incorrect_op_sims = extract_token_similarities(
                softmax_sims, incorrect_op_token_ids
            )
            avg_incorrect_sims = incorrect_op_sims.mean(dim=-1)  # [num_latents]

            for latent_idx in range(len(avg_incorrect_sims)):
                latent_position_incorrect_ops[latent_idx].append(
                    avg_incorrect_sims[latent_idx].item()
                )

        # Extract similarities for incorrect operations (verbal)
        if len(incorrect_op_verbal_token_ids) > 0:
            incorrect_op_verbal_sims = extract_token_similarities(
                softmax_sims, incorrect_op_verbal_token_ids
            )
            avg_incorrect_verbal_sims = incorrect_op_verbal_sims.mean(
                dim=-1
            )  # [num_latents]

            for latent_idx in range(len(avg_incorrect_verbal_sims)):
                latent_position_incorrect_ops_verbal[latent_idx].append(
                    avg_incorrect_verbal_sims[latent_idx].item()
                )

        # Baseline: Similarity to random tokens
        random_sims = extract_token_similarities(softmax_sims, random_token_ids)
        avg_random_sims = random_sims.mean(dim=-1)  # [num_latents]

        for latent_idx in range(len(avg_random_sims)):
            latent_position_baseline_random[latent_idx].append(
                avg_random_sims[latent_idx].item()
            )

        samples_processed += 1

    except Exception as e:
        print(f"\nError processing sample {idx}: {e}")
        continue

print(f"\nProcessed {samples_processed} samples")
print(f"Samples with intermediate operations: {samples_with_operations}")

# %%
# ==================== COMPUTE AVERAGES ====================
print("\nComputing average similarities across all samples...")

# Average and std dev across all samples for each latent position
avg_correct_per_latent = []
std_correct_per_latent = []
avg_correct_verbal_per_latent = []
std_correct_verbal_per_latent = []
avg_incorrect_per_latent = []
std_incorrect_per_latent = []
avg_incorrect_verbal_per_latent = []
std_incorrect_verbal_per_latent = []
avg_random_per_latent = []
std_random_per_latent = []

for latent_idx in range(NUM_LATENT_ITERATIONS + 1):
    if len(latent_position_correct_ops[latent_idx]) > 0:
        avg_correct_per_latent.append(np.mean(latent_position_correct_ops[latent_idx]))
        std_correct_per_latent.append(np.std(latent_position_correct_ops[latent_idx]))
    else:
        avg_correct_per_latent.append(0)
        std_correct_per_latent.append(0)

    if len(latent_position_correct_ops_verbal[latent_idx]) > 0:
        avg_correct_verbal_per_latent.append(
            np.mean(latent_position_correct_ops_verbal[latent_idx])
        )
        std_correct_verbal_per_latent.append(
            np.std(latent_position_correct_ops_verbal[latent_idx])
        )
    else:
        avg_correct_verbal_per_latent.append(0)
        std_correct_verbal_per_latent.append(0)

    if len(latent_position_incorrect_ops[latent_idx]) > 0:
        avg_incorrect_per_latent.append(
            np.mean(latent_position_incorrect_ops[latent_idx])
        )
        std_incorrect_per_latent.append(
            np.std(latent_position_incorrect_ops[latent_idx])
        )
    else:
        avg_incorrect_per_latent.append(0)
        std_incorrect_per_latent.append(0)

    if len(latent_position_incorrect_ops_verbal[latent_idx]) > 0:
        avg_incorrect_verbal_per_latent.append(
            np.mean(latent_position_incorrect_ops_verbal[latent_idx])
        )
        std_incorrect_verbal_per_latent.append(
            np.std(latent_position_incorrect_ops_verbal[latent_idx])
        )
    else:
        avg_incorrect_verbal_per_latent.append(0)
        std_incorrect_verbal_per_latent.append(0)

    if len(latent_position_baseline_random[latent_idx]) > 0:
        avg_random_per_latent.append(
            np.mean(latent_position_baseline_random[latent_idx])
        )
        std_random_per_latent.append(
            np.std(latent_position_baseline_random[latent_idx])
        )
    else:
        avg_random_per_latent.append(0)
        std_random_per_latent.append(0)

print("Average similarities and standard deviations computed")

# %%
# ==================== SAVE RESULTS ====================
print(f"\nSaving results to {OUTPUT_DATA}...")

results = {
    "num_samples": samples_processed,
    "num_latent_iterations": NUM_LATENT_ITERATIONS,
    "operations": OPERATIONS,
    "operation_words": OPERATION_WORDS,
    "operation_token_ids": {
        op: int(tid) for op, tid in operation_token_ids.items()
    },  # Convert to int for JSON
    "operation_word_token_ids": {
        op: [int(tid) for tid in tids] for op, tids in operation_word_token_ids.items()
    },
    "avg_similarity_to_correct_ops": avg_correct_per_latent,
    "std_similarity_to_correct_ops": std_correct_per_latent,
    "avg_similarity_to_correct_ops_verbal": avg_correct_verbal_per_latent,
    "std_similarity_to_correct_ops_verbal": std_correct_verbal_per_latent,
    "avg_similarity_to_incorrect_ops": avg_incorrect_per_latent,
    "std_similarity_to_incorrect_ops": std_incorrect_per_latent,
    "avg_similarity_to_incorrect_ops_verbal": avg_incorrect_verbal_per_latent,
    "std_similarity_to_incorrect_ops_verbal": std_incorrect_verbal_per_latent,
    "avg_similarity_to_random_tokens": avg_random_per_latent,
    "std_similarity_to_random_tokens": std_random_per_latent,
    "raw_data": {
        "correct_op_similarities": [
            latent_position_correct_ops[i]
            for i in range(len(latent_position_correct_ops))
        ],
        "correct_op_verbal_similarities": [
            latent_position_correct_ops_verbal[i]
            for i in range(len(latent_position_correct_ops_verbal))
        ],
        "incorrect_op_similarities": [
            latent_position_incorrect_ops[i]
            for i in range(len(latent_position_incorrect_ops))
        ],
        "incorrect_op_verbal_similarities": [
            latent_position_incorrect_ops_verbal[i]
            for i in range(len(latent_position_incorrect_ops_verbal))
        ],
        "random_similarities": [
            latent_position_baseline_random[i]
            for i in range(len(latent_position_baseline_random))
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
avg_correct_np = np.array(avg_correct_per_latent)
std_correct_np = np.array(std_correct_per_latent)
avg_correct_verbal_np = np.array(avg_correct_verbal_per_latent)
std_correct_verbal_np = np.array(std_correct_verbal_per_latent)
avg_incorrect_np = np.array(avg_incorrect_per_latent)
std_incorrect_np = np.array(std_incorrect_per_latent)
avg_incorrect_verbal_np = np.array(avg_incorrect_verbal_per_latent)
std_incorrect_verbal_np = np.array(std_incorrect_verbal_per_latent)
avg_random_np = np.array(avg_random_per_latent)
std_random_np = np.array(std_random_per_latent)

# Plot correct operations (symbolic)
plt.plot(
    x_positions,
    avg_correct_np,
    marker="o",
    linewidth=3,
    markersize=10,
    label="Correct Ops (Symbolic)",
    color="blue",
    zorder=3,
)
plt.fill_between(
    x_positions,
    avg_correct_np - std_correct_np,
    avg_correct_np + std_correct_np,
    color="blue",
    alpha=0.2,
    zorder=1,
)

# Plot correct operations (verbal)
plt.plot(
    x_positions,
    avg_correct_verbal_np,
    marker="s",
    linewidth=3,
    markersize=10,
    label="Correct Ops (Verbal)",
    color="cyan",
    linestyle="-",
    zorder=3,
)
plt.fill_between(
    x_positions,
    avg_correct_verbal_np - std_correct_verbal_np,
    avg_correct_verbal_np + std_correct_verbal_np,
    color="cyan",
    alpha=0.2,
    zorder=1,
)

# Plot incorrect operations (symbolic)
plt.plot(
    x_positions,
    avg_incorrect_np,
    marker="D",
    linewidth=3,
    markersize=10,
    label="Incorrect Ops (Symbolic)",
    color="red",
    linestyle="--",
    zorder=3,
)
plt.fill_between(
    x_positions,
    avg_incorrect_np - std_incorrect_np,
    avg_incorrect_np + std_incorrect_np,
    color="red",
    alpha=0.2,
    zorder=1,
)

# Plot incorrect operations (verbal)
plt.plot(
    x_positions,
    avg_incorrect_verbal_np,
    marker="v",
    linewidth=3,
    markersize=10,
    label="Incorrect Ops (Verbal)",
    color="orange",
    linestyle="--",
    zorder=3,
)
plt.fill_between(
    x_positions,
    avg_incorrect_verbal_np - std_incorrect_verbal_np,
    avg_incorrect_verbal_np + std_incorrect_verbal_np,
    color="orange",
    alpha=0.2,
    zorder=1,
)

# Plot random baseline
plt.plot(
    x_positions,
    avg_random_np,
    marker="^",
    linewidth=3,
    markersize=10,
    label="Random Tokens",
    color="gray",
    linestyle=":",
    zorder=3,
)
plt.fill_between(
    x_positions,
    avg_random_np - std_random_np,
    avg_random_np + std_random_np,
    color="gray",
    alpha=0.15,
    zorder=1,
)

plt.xlabel("Latent Vector Index", fontsize=16)
plt.ylabel("Average Similarity", fontsize=16)
plt.title(
    "Latent Vector Similarity to Mathematical Operation Tokens",
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
print("\n" + "=" * 120)
print("SUMMARY")
print("=" * 120)
print(f"Samples processed: {samples_processed}")
print(f"Latent iterations: {NUM_LATENT_ITERATIONS}")
print(f"Operations analyzed: {', '.join(OPERATIONS)}")
print()
print("Average Softmax Probabilities (Mean ± Std Dev, values in [0, 1]):")
print("-" * 120)
print(
    f"{'Latent':<8} {'Correct (Sym)':<22} {'Correct (Verb)':<22} "
    f"{'Incorrect (Sym)':<22} {'Incorrect (Verb)':<22} {'Random':<22}"
)
print("-" * 120)
for i in range(NUM_LATENT_ITERATIONS + 1):
    print(
        f"{i:<8} "
        f"{avg_correct_per_latent[i]:.4f}±{std_correct_per_latent[i]:.4f}    "
        f"{avg_correct_verbal_per_latent[i]:.4f}±{std_correct_verbal_per_latent[i]:.4f}    "
        f"{avg_incorrect_per_latent[i]:.4f}±{std_incorrect_per_latent[i]:.4f}    "
        f"{avg_incorrect_verbal_per_latent[i]:.4f}±{std_incorrect_verbal_per_latent[i]:.4f}    "
        f"{avg_random_per_latent[i]:.4f}±{std_random_per_latent[i]:.4f}"
    )
print("-" * 120)

# Compute differences
print("\nDifference: Correct (Symbolic) vs Incorrect (Symbolic):")
for i in range(NUM_LATENT_ITERATIONS + 1):
    diff = avg_correct_per_latent[i] - avg_incorrect_per_latent[i]
    print(f"  Latent {i}: {diff:+.6f}")

print("\nDifference: Correct (Verbal) vs Incorrect (Verbal):")
for i in range(NUM_LATENT_ITERATIONS + 1):
    diff = avg_correct_verbal_per_latent[i] - avg_incorrect_verbal_per_latent[i]
    print(f"  Latent {i}: {diff:+.6f}")

print("\nDifference: Correct (Symbolic) vs Correct (Verbal):")
for i in range(NUM_LATENT_ITERATIONS + 1):
    diff = avg_correct_per_latent[i] - avg_correct_verbal_per_latent[i]
    print(f"  Latent {i}: {diff:+.6f}")

print("\nDifference: Correct (Symbolic) vs Random Tokens:")
for i in range(NUM_LATENT_ITERATIONS + 1):
    diff = avg_correct_per_latent[i] - avg_random_per_latent[i]
    print(f"  Latent {i}: {diff:+.6f}")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)

# %%
