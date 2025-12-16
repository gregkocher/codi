# ABOUTME: Experiment to swap individual latent vectors between different prompts.
# ABOUTME: Tests if swapping one latent vector changes the model's reasoning and output.
# %%
import torch
import transformers
from dotenv import load_dotenv

from src.model import CODI

load_dotenv()

# %%
# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================
# Model configuration
MODEL_CHECKPOINT = "bcywinski/codi_llama1b-answer_only"
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
CHECKPOINT_SAVE_PATH = "./checkpoints/bcywinski/codi_llama1b-answer_only"

# Generation parameters
NUM_LATENT_ITERATIONS = 6
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.1
TOP_K = 40
TOP_P = 0.95
GREEDY = True

# Prompt template parameters
# First prompt
X1 = 4
Y1 = 2
Z1 = 4

# Second prompt (change ONE value)
X2 = 3
Y2 = 4
Z2 = 4

# Third prompt - use first prompt values, will swap latent vectors
X3 = X1
Y3 = Y1
Z3 = Z1

# Which latent vector index to swap (0 to NUM_LATENT_ITERATIONS)
# 0 = initial latent, 1-6 = iteration latents
LATENT_INDEX_TO_SWAP = 2

# =============================================================================
# LOAD MODEL
# =============================================================================
# %%
print("Loading model...")
model = CODI.from_pretrained(
    checkpoint_path=MODEL_CHECKPOINT,
    model_name_or_path=BASE_MODEL,
    lora_r=128,
    lora_alpha=32,
    num_latent=NUM_LATENT_ITERATIONS,
    use_prj=True,
    device="cuda",
    dtype="bfloat16",
    strict=False,
    checkpoint_save_path=CHECKPOINT_SAVE_PATH,
    remove_eos=True,
    full_precision=True,
)
print("Model loaded!")

# %%
# Setup tokenizer
tokenizer = model.tokenizer

if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token_id = model.pad_token_id
tokenizer.add_special_tokens({"additional_special_tokens": ["<|bocot|>", "<|eocot|>"]})
tokenizer.bocot_id = tokenizer.convert_tokens_to_ids("<|bocot|>")
tokenizer.eocot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")

# =============================================================================
# INFERENCE 1: First prompt
# =============================================================================
# %%
print("\n" + "=" * 80)
print("INFERENCE 1: First prompt")
print("=" * 80)

prompt1 = f"""A team starts with {X1} members. They recruit {Y1} new members. Then each current member recruits {Z1} additional people. How many people are there now on the team? Give the answer only and nothing else."""
print(f"Prompt 1: {prompt1}")

inputs1 = tokenizer(prompt1, return_tensors="pt", padding=True)
input_ids1 = inputs1["input_ids"].to(model.codi.device)
print(input_ids1[0].shape)
attention_mask1 = inputs1["attention_mask"].to(model.codi.device)

# %%
output1 = model.generate(
    input_ids=input_ids1,
    attention_mask=attention_mask1,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    num_latent_iterations=NUM_LATENT_ITERATIONS,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    top_p=TOP_P,
    greedy=GREEDY,
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
generated_text1 = tokenizer.decode(output1["sequences"][0], skip_special_tokens=False)
print(f"\nGenerated 1: {generated_text1}")

latent_vectors1 = output1["latent_vectors"]
print(f"Number of latent vectors from inference 1: {len(latent_vectors1)}")
print(f"Shape of each latent vector: {latent_vectors1[0].shape}")

# %%
# Analyze latent vector similarities to token embeddings for Inference 1
print("\n" + "-" * 80)
print("TOP 10 TOKENS FOR EACH LATENT VECTOR (INFERENCE 1)")
print("-" * 80)

latent_vectors1_stacked = torch.stack(latent_vectors1).squeeze(1).squeeze(1).to("cpu")
embed_matrix = model.codi.model.model.embed_tokens.weight.to("cpu")
sims1 = latent_vectors1_stacked @ embed_matrix.T
sims1 = sims1.softmax(dim=-1)

k = 10
for i in range(sims1.shape[0]):
    topk = sims1[i].topk(k, dim=-1)
    topk_indices = topk.indices.tolist()
    topk_values = topk.values.tolist()
    print(f"\nLatent vector #{i}:")
    print(f"{'Rank':<5} {'Token':<20}{'Similarity':<10}")
    print("-" * 50)
    for rank, (j, value) in enumerate(zip(topk_indices, topk_values), 1):
        token_str = tokenizer.convert_ids_to_tokens([j])[0]
        print(f"{rank:<5} {token_str:<20} {value:<10.6f}")
    print("-" * 50)

# =============================================================================
# INFERENCE 2: Second prompt (one value changed)
# =============================================================================
# %%
print("\n" + "=" * 80)
print("INFERENCE 2: Second prompt (one value changed)")
print("=" * 80)

prompt2 = f"""A team starts with {X2} members. They recruit {Y2} new members. Then each current member recruits {Z2} additional people. How many people are there now on the team? Give the answer only and nothing else."""
print(f"Prompt 2: {prompt2}")

inputs2 = tokenizer(prompt2, return_tensors="pt", padding=True)
input_ids2 = inputs2["input_ids"].to(model.codi.device)
print(input_ids2[0].shape)
attention_mask2 = inputs2["attention_mask"].to(model.codi.device)

# %%
output2 = model.generate(
    input_ids=input_ids2,
    attention_mask=attention_mask2,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    num_latent_iterations=NUM_LATENT_ITERATIONS,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    top_p=TOP_P,
    greedy=GREEDY,
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
generated_text2 = tokenizer.decode(output2["sequences"][0], skip_special_tokens=False)
print(f"\nGenerated 2: {generated_text2}")

latent_vectors2 = output2["latent_vectors"]
print(f"Number of latent vectors from inference 2: {len(latent_vectors2)}")
print(f"Shape of each latent vector: {latent_vectors2[0].shape}")

# %%
# Analyze latent vector similarities to token embeddings for Inference 2
print("\n" + "-" * 80)
print("TOP 10 TOKENS FOR EACH LATENT VECTOR (INFERENCE 2)")
print("-" * 80)

latent_vectors2_stacked = torch.stack(latent_vectors2).squeeze(1).squeeze(1).to("cpu")
sims2 = latent_vectors2_stacked @ embed_matrix.T
sims2 = sims2.softmax(dim=-1)

k = 10
for i in range(sims2.shape[0]):
    topk = sims2[i].topk(k, dim=-1)
    topk_indices = topk.indices.tolist()
    topk_values = topk.values.tolist()
    print(f"\nLatent vector #{i}:")
    print(f"{'Rank':<5} {'Token':<20}{'Similarity':<10}")
    print("-" * 50)
    for rank, (j, value) in enumerate(zip(topk_indices, topk_values), 1):
        token_str = tokenizer.convert_ids_to_tokens([j])[0]
        print(f"{rank:<5} {token_str:<20} {value:<10.6f}")
    print("-" * 50)

# =============================================================================
# PREPARE SWAPPED LATENT VECTORS
# =============================================================================
# %%
print("\n" + "=" * 80)
print("PREPARING SWAPPED LATENT VECTORS")
print("=" * 80)

# Start with all latent vectors from inference 1
swapped_latent_vectors = [lv.clone() for lv in latent_vectors1]

# Swap the specified latent vector with the one from inference 2
print(
    f"Swapping latent vector at index {LATENT_INDEX_TO_SWAP} from inference 2 into inference 1's vectors"
)
# swapped_latent_vectors[LATENT_INDEX_TO_SWAP] = latent_vectors2[
#     LATENT_INDEX_TO_SWAP
# ].clone()

print(f"Total latent vectors prepared: {len(swapped_latent_vectors)}")


# =============================================================================
# INFERENCE 3: Using swapped latent vectors
# =============================================================================
# %%
print("\n" + "=" * 80)
print("INFERENCE 3: Using swapped latent vectors")
print("=" * 80)

prompt3 = """A team starts with  X members. They recruit  Y new members. Then each current member recruits  Z additional people. How many people are there now on the team? Give the answer only and nothing else."""
print(f"Prompt 3: {prompt3}")
print(f"(Should be same as Prompt 1: {prompt3 == prompt1})")

inputs3 = tokenizer(prompt3, return_tensors="pt", padding=True)
print(inputs3["input_ids"][0].shape)
input_ids3 = inputs3["input_ids"].to(model.codi.device)
attention_mask3 = inputs3["attention_mask"].to(model.codi.device)

# %%
output3 = model.generate(
    input_ids=input_ids3,
    attention_mask=attention_mask3,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    num_latent_iterations=NUM_LATENT_ITERATIONS,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    top_p=TOP_P,
    greedy=GREEDY,
    return_latent_vectors=True,
    remove_eos=False,
    output_attentions=False,
    skip_thinking=False,
    verbalize_cot=False,
    output_hidden_states=True,
    sot_token=tokenizer.convert_tokens_to_ids("<|bocot|>"),
    eot_token=tokenizer.convert_tokens_to_ids("<|eocot|>"),
    latent_vectors_override=swapped_latent_vectors,  # Use the swapped latents!
)

# %%
generated_text3 = tokenizer.decode(output3["sequences"][0], skip_special_tokens=False)
print(f"\nGenerated 3: {generated_text3}")

# %%
# Analyze latent vector similarities to token embeddings for Inference 3
print("\n" + "-" * 80)
print("TOP 10 TOKENS FOR EACH LATENT VECTOR (INFERENCE 3 - WITH SWAPPED LATENT)")
print("-" * 80)

latent_vectors3 = output3["latent_vectors"]
latent_vectors3_stacked = torch.stack(latent_vectors3).squeeze(1).squeeze(1).to("cpu")
sims3 = latent_vectors3_stacked @ embed_matrix.T
sims3 = sims3.softmax(dim=-1)

k = 10
for i in range(sims3.shape[0]):
    topk = sims3[i].topk(k, dim=-1)
    topk_indices = topk.indices.tolist()
    topk_values = topk.values.tolist()
    marker = " <-- SWAPPED FROM INFERENCE 2" if i == LATENT_INDEX_TO_SWAP else ""
    print(f"\nLatent vector #{i}{marker}:")
    print(f"{'Rank':<5} {'Token':<20}{'Similarity':<10}")
    print("-" * 50)
    for rank, (j, value) in enumerate(zip(topk_indices, topk_values), 1):
        token_str = tokenizer.convert_ids_to_tokens([j])[0]
        print(f"{rank:<5} {token_str:<20} {value:<10.6f}")
    print("-" * 50)

# =============================================================================
# COMPARE RESULTS
# =============================================================================
# %%
print("\n" + "=" * 80)
print("EXPERIMENT SUMMARY")
print("=" * 80)

print(f"\nPrompt 1 (X={X1}, Y={Y1}, Z={Z1}):")
print(f"  Input: {prompt1}")
print(f"  Output: {generated_text1}")

print(f"\nPrompt 2 (X={X2}, Y={Y2}, Z={Z2}):")
print(f"  Input: {prompt2}")
print(f"  Output: {generated_text2}")

print(
    f"\nPrompt 3 (X={X3}, Y={Y3}, Z={Z3}) with swapped latent #{LATENT_INDEX_TO_SWAP}:"
)
print(f"  Input: {prompt3}")
print(f"  Output: {generated_text3}")

print(f"\n{'=' * 80}")
print("ANALYSIS")
print("=" * 80)
print(f"Latent vector #{LATENT_INDEX_TO_SWAP} was swapped from Prompt 2 into Prompt 3")
print(f"Does swapping change the output?")
print(f"  Inference 1 == Inference 3: {generated_text1 == generated_text3}")
print(f"  Inference 2 == Inference 3: {generated_text2 == generated_text3}")

if generated_text1 != generated_text3:
    print(f"\n✓ Swapping latent vector #{LATENT_INDEX_TO_SWAP} CHANGED the output!")
else:
    print(
        f"\n✗ Swapping latent vector #{LATENT_INDEX_TO_SWAP} did NOT change the output"
    )

# %%
# Compute cosine similarities between latent vectors
print("\n" + "=" * 80)
print("LATENT VECTOR SIMILARITIES (INFERENCE 1 vs INFERENCE 2)")
print("=" * 80)

for i in range(len(latent_vectors1)):
    # Flatten and normalize for cosine similarity
    lv1_flat = latent_vectors1[i].flatten()
    lv2_flat = latent_vectors2[i].flatten()

    lv1_norm = lv1_flat / lv1_flat.norm()
    lv2_norm = lv2_flat / lv2_flat.norm()

    cosine_sim = (lv1_norm @ lv2_norm).item()

    marker = " <-- SWAPPED" if i == LATENT_INDEX_TO_SWAP else ""
    print(f"Latent #{i}: Cosine similarity = {cosine_sim:.6f}{marker}")

# %%
# Compare top tokens for the swapped latent vector across all three inferences
print("\n" + "=" * 80)
print(
    f"DETAILED COMPARISON: LATENT VECTOR #{LATENT_INDEX_TO_SWAP} ACROSS ALL INFERENCES"
)
print("=" * 80)

k_compare = 10
print(
    f"\n{'Rank':<5} {'Inference 1':<25} {'Inference 2':<25} {'Inference 3 (Swapped)':<25}"
)
print("-" * 85)

topk1 = sims1[LATENT_INDEX_TO_SWAP].topk(k_compare, dim=-1)
topk2 = sims2[LATENT_INDEX_TO_SWAP].topk(k_compare, dim=-1)
topk3 = sims3[LATENT_INDEX_TO_SWAP].topk(k_compare, dim=-1)

for rank in range(k_compare):
    tok1 = tokenizer.convert_ids_to_tokens([topk1.indices[rank].item()])[0]
    tok2 = tokenizer.convert_ids_to_tokens([topk2.indices[rank].item()])[0]
    tok3 = tokenizer.convert_ids_to_tokens([topk3.indices[rank].item()])[0]

    val1 = topk1.values[rank].item()
    val2 = topk2.values[rank].item()
    val3 = topk3.values[rank].item()

    print(
        f"{rank + 1:<5} {tok1:<15} {val1:>8.6f}  {tok2:<15} {val2:>8.6f}  {tok3:<15} {val3:>8.6f}"
    )

print("-" * 85)
print(
    f"\nNote: Inference 3's latent #{LATENT_INDEX_TO_SWAP} should match Inference 2's"
)

# %%
