# ABOUTME: Analyzes similarity between token embeddings and model activations across layers
# ABOUTME: Compares target token similarity against baseline of random number tokens

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from src.model import CODI

load_dotenv()

# %%
# Configuration
X = 3
Y = 2
Z = 4


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
# Load tokenizer
tokenizer = model.tokenizer

if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token_id = model.pad_token_id
tokenizer.add_special_tokens({"additional_special_tokens": ["<|bocot|>", "<|eocot|>"]})
tokenizer.bocot_id = tokenizer.convert_tokens_to_ids("<|bocot|>")
tokenizer.eocot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")

# %%
# Create prompt
prompt = f"""A team starts with {X} members. They recruit {Y} new members. Then each current member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else."""
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
input_ids = inputs["input_ids"].to(model.codi.device)
attention_mask = inputs["attention_mask"].to(model.codi.device)

# %%
# Generate with hidden states
output = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    tokenizer=tokenizer,
    max_new_tokens=256,
    num_latent_iterations=6,
    temperature=0.1,
    top_k=40,
    top_p=0.95,
    greedy=True,
    return_latent_vectors=True,
    remove_eos=False,
    output_attentions=False,
    skip_thinking=False,
    verbalize_cot=False,
    output_hidden_states=True,
    sot_token=tokenizer.convert_tokens_to_ids("<|bocot|>"),
    eot_token=tokenizer.convert_tokens_to_ids("<|eocot|>"),
)

# %%
# Extract and process hidden states
hs = output["hidden_states"].to("cpu")
print(f"Hidden states shape: {hs.shape}")

# Get embedding matrix
embedding_matrix = model.codi.base_model.model.model.embed_tokens.weight
print(f"Embedding matrix shape: {embedding_matrix.shape}")

# %%
target_token_str = (
    "5"  # Token to analyze (appears as intermediate result: 3+2=5, 5*4=20)
)
num_baseline_samples = 1000  # Number of random number tokens for baseline
# Get target token ID
target_token_id = tokenizer.convert_tokens_to_ids(target_token_str)
if (
    target_token_id is None
    or target_token_id < 0
    or target_token_id >= embedding_matrix.shape[0]
):
    raise ValueError(
        f"Token '{target_token_str}' not found or out of valid embedding range."
    )

print(f"Target token: '{target_token_str}' (id: {target_token_id})")

# %%
# Prepare hidden states for similarity calculation
if hs.ndim == 4:
    hs2d = hs[:, 0, :, :]  # [num_layers, num_tokens, hidden_dim]
else:
    hs2d = hs

hs2d = hs2d.float()

# %%
# Calculate similarity using softmax over entire vocabulary
# Methodology: For each activation (layer, position):
#   1. Compute dot product with ALL token embeddings
#   2. Apply softmax to get probability distribution over vocabulary
#   3. Extract probability for target token (or average for baseline tokens)
# This ensures all values are between 0 and 1, representing how much
# each activation "looks like" a specific token in probability space
embedding_matrix_cpu = embedding_matrix.cpu().float()

# Calculate similarity to ALL tokens in the vocabulary
# Shape: [num_layers, num_tokens, vocab_size]
print("Calculating similarities to all tokens in vocabulary...")
all_similarities = torch.matmul(hs2d, embedding_matrix_cpu.T)
print(f"All similarities shape: {all_similarities.shape}")

# Apply softmax to normalize to probability distribution
print("Applying softmax normalization...")
all_similarities_softmax = torch.softmax(all_similarities, dim=-1)

# Extract probability for target token
target_cosine_sim = all_similarities_softmax[:, :, target_token_id].numpy()
print(f"Target similarity shape: {target_cosine_sim.shape}")

# %%
# Generate baseline: average similarity with random 1000 number tokens
# Get token IDs for numbers 0-999
number_tokens = []
for i in range(1000):
    token_str = str(i)
    token_id = tokenizer.convert_tokens_to_ids(token_str)
    if token_id is not None and token_id >= 0 and token_id < embedding_matrix.shape[0]:
        number_tokens.append(token_id)

print(f"Found {len(number_tokens)} valid number tokens")

# %%
# Extract probabilities for all baseline tokens
print("Extracting baseline token probabilities...")
baseline_token_ids = torch.tensor(number_tokens)
baseline_similarities = all_similarities_softmax[:, :, baseline_token_ids].numpy()
# Shape: [num_layers, num_positions, num_baseline_tokens]

# Average across baseline tokens
baseline_avg = baseline_similarities.mean(axis=-1)  # [num_layers, num_positions]
print(f"Baseline average similarity shape: {baseline_avg.shape}")

# %%
# Calculate difference: target similarity - baseline
difference = target_cosine_sim - baseline_avg


# %%
# Values are already between 0 and 1 due to softmax normalization
# No additional normalization needed for target and baseline
# Difference can be negative, so we keep it as-is

# %%
# Prepare token labels for x-axis
num_layers, num_tokens = target_cosine_sim.shape
token_ids = input_ids[0, :num_tokens].tolist()
token_strs = tokenizer.convert_ids_to_tokens(token_ids)

# %%
# Plot target token probability
plt.figure(figsize=(min(12, 0.75 * num_tokens), 7))
im = plt.imshow(target_cosine_sim, aspect="auto", cmap="viridis", vmin=0, vmax=1)
plt.colorbar(im, label="Softmax probability")
plt.ylabel("Layer")
plt.xlabel("Token Position")
plt.title(f"Target token '{target_token_str}' probability (softmax over vocab)")
plt.xticks(np.arange(len(token_strs)), token_strs, rotation=90, fontsize=8)
plt.tight_layout()
plt.savefig("token_similarity_with_baseline.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nPlot saved to 'token_similarity_with_baseline.png'")
print(f"\nPrompt values: X={X}, Y={Y}, Z={Z}")
print(f"Expected intermediate: {X}+{Y}={X + Y}, then {X + Y}*{Z}={(X + Y) * Z}")

# %%
# Print statistics
print(f"\nTarget token probability statistics:")
print(f"  Min: {target_cosine_sim.min():.6f}")
print(f"  Max: {target_cosine_sim.max():.6f}")
print(f"  Mean: {target_cosine_sim.mean():.6f}")

print(f"\nBaseline average probability statistics:")
print(f"  Min: {baseline_avg.min():.6f}")
print(f"  Max: {baseline_avg.max():.6f}")
print(f"  Mean: {baseline_avg.mean():.6f}")

print(f"\nDifference (target - baseline) statistics:")
print(f"  Min: {difference.min():.6f}")
print(f"  Max: {difference.max():.6f}")
print(f"  Mean: {difference.mean():.6f}")

# %%
