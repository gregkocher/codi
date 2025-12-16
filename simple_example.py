# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv

from src.model import CODI

load_dotenv()
# %%
model = CODI.from_pretrained(
    checkpoint_path="bcywinski/codi_llama1b-answer_only",  # HF checkpoint ID
    model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",  # HF base model ID
    lora_r=128,
    lora_alpha=32,
    num_latent=6,  # Number of latent reasoning steps
    use_prj=True,  # Whether model uses projection layer
    device="cuda",
    dtype="bfloat16",
    strict=False,
    # Optional: specify where to save the checkpoint (default: ./checkpoints/{name})
    checkpoint_save_path="./checkpoints/bcywinski/codi_llama1b-answer_only",
    remove_eos=False,
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
X = 3
Y = 2
Z = 4
prompt = f"""A team starts with {X} members. {Y} members leave the team. Then each remaining member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else."""
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
input_ids = inputs["input_ids"].to(model.codi.device)
attention_mask = inputs["attention_mask"].to(model.codi.device)

# %%
skip_thinking = False
verbalize_cot = False
output = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    tokenizer=tokenizer,
    max_new_tokens=256,
    num_latent_iterations=6,  # How many latent reasoning steps
    temperature=0.1,
    top_k=40,
    top_p=0.95,
    greedy=True,
    return_latent_vectors=True,  # Get the latent reasoning vectors
    remove_eos=False,
    output_attentions=False,  # Get attention weights
    skip_thinking=skip_thinking,
    verbalize_cot=verbalize_cot,
    output_hidden_states=True,
    sot_token=tokenizer.convert_tokens_to_ids("<|bocot|>"),
    eot_token=tokenizer.convert_tokens_to_ids("<|eocot|>"),
)

# %%
# Decode and print output
generated_text = tokenizer.decode(output["sequences"][0], skip_special_tokens=False)
print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")
# %%
hs = output["hidden_states"].to("cpu")
hs.shape
# %%
latent_vectors = torch.stack(output["latent_vectors"]).squeeze(1).squeeze(1).to("cpu")
latent_vectors.shape
# %%
embed_matrix = model.codi.model.model.embed_tokens.weight.to("cpu")
# %%
sims = latent_vectors @ embed_matrix.T
sims.shape
# %%
sims = sims.softmax(dim=-1)
# %%
k = 10
for i in range(sims.shape[0]):
    topk = sims[i].topk(k, dim=-1)
    topk_indices = topk.indices.tolist()
    topk_values = topk.values.tolist()
    print(f"\nLatent vector #{i + 1}:")
    print(f"{'Rank':<5} {'Token':<20}{'Similarity':<10}")
    print("-" * 50)
    for rank, (j, value) in enumerate(zip(topk_indices, topk_values), 1):
        token_str = tokenizer.convert_ids_to_tokens([j])[0]
        print(f"{rank:<5} {token_str:<20} {value:<10.6f}")
    print("-" * 50)
# %%
# Select which latent vector to analyze
selected_latent_idx = 0  # change if you want a different latent vector
selected_latent_vector = latent_vectors[selected_latent_idx]  # shape: [2048]

# output["hidden_states"] is (seq_len, n_layers, hidden_dim) for batch size 1.
hs = output["hidden_states"].to("cpu")
hs_by_layer = hs.permute(1, 0, 2)  # (num_layers, num_tokens, hidden_dim)

# Normalize both hs and selected latent vector for cosine similarity
hs_norm = hs_by_layer / hs_by_layer.norm(dim=-1, keepdim=True)
latent_norm = selected_latent_vector / selected_latent_vector.norm()  # (hidden_dim,)

# Compute similarity: for each [layer, token], dot with latent vector
# Output shape: (num_layers, num_tokens)
similarity = torch.einsum("ltd,d->lt", hs_norm, latent_norm).cpu().float().numpy()

# Only consider the last layer
last_layer_sim = similarity[-1]  # shape: (num_tokens,)

# Get corresponding tokens as strings for y-axis
num_tokens = last_layer_sim.shape[0]
token_ids = input_ids[0, :num_tokens].tolist()
token_strs = tokenizer.convert_ids_to_tokens(token_ids)

# Plotting similarities for the last layer, y-axis as tokens
plt.figure(figsize=(10, max(3, 0.35 * num_tokens)))
y_pos = range(num_tokens)
plt.barh(y_pos, last_layer_sim, align="center", color="dodgerblue")
plt.yticks(y_pos, token_strs, fontsize=20)
plt.xlabel("Cosine similarity")
plt.ylabel("Token")
plt.title(f"Cosine similarity (last layer): latent vector #{selected_latent_idx + 1}")
plt.gca().invert_yaxis()  # Highest similarity on top
plt.tight_layout()
plt.show()
# %%
# Only use similarities from the last layer
last_layer_sim = similarity[-1]  # shape: (num_tokens,)

# Sort the similarities and their indices in descending order
sorted_indices = last_layer_sim.argsort()[::-1]
sorted_similarities = last_layer_sim[sorted_indices]

# Print similarities and corresponding tokens
print(f"{'Rank':<5} {'TokenIdx':<8} {'Token':<20} {'Similarity':<10}")
print("-" * 50)
for rank, (token_idx, sim_value) in enumerate(
    zip(sorted_indices, sorted_similarities), 1
):
    token_id = (
        input_ids[0, token_idx].item()
        if (input_ids is not None and token_idx < input_ids.shape[1])
        else None
    )
    token_str = (
        tokenizer.convert_ids_to_tokens([token_id])[0] if token_id is not None else "?"
    )
    print(f"{rank:<5} {token_idx:<8} {token_str:<20} {sim_value:<10.6f}")
print("-" * 50)
# Choose the specific target token (string)
target_token_str = "5"

# Get token id for the specific target token
target_token_id = tokenizer.convert_tokens_to_ids(target_token_str)
if target_token_id is None or target_token_id < 0 or target_token_id >= 128259:
    raise ValueError(
        f"Token '{target_token_str}' not found or out of valid embedding range."
    )

# Get embedding matrix [vocab_size, hidden_dim]
embedding_matrix = (
    model.codi.base_model.model.model.embed_tokens.weight
)  # [128259, 2048]

# Get target token embedding and normalize
token_embedding = embedding_matrix[target_token_id].cpu().float()  # [2048]
token_embedding_norm = token_embedding / token_embedding.norm()

# Prepare hs2d [num_layers, num_tokens, hidden_dim] for cosine computation
if hs_norm.ndim == 4:
    hs2d = hs_norm[:, 0, :, :]
else:
    hs2d = hs_norm
hs2d = hs2d.float()

# Compute similarity of activations to target token embedding
target_cosine_sim = (
    torch.einsum("ltd,d->lt", hs2d, token_embedding_norm).cpu().float().numpy()
)  # [num_layers, num_tokens]

# --- BASELINE: Compute similarity to 1000 random number tokens ---
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

if len(valid_number_token_ids) < num_to_find:
    raise ValueError(f"Only found {len(valid_number_token_ids)} valid number tokens.")

baseline_token_ids = valid_number_token_ids[:num_to_find]

# Compute baseline similarities: average over 1000 number tokens
# Get [num_baseline, hidden_dim] matrix of embeddings
baseline_embeds = embedding_matrix[baseline_token_ids, :].cpu().float()
baseline_embeds_norm = baseline_embeds / baseline_embeds.norm(dim=1, keepdim=True)

# Compute for each layer and token: [num_layers, num_tokens, hidden_dim] x [num_baseline, hidden_dim]
# Result: [num_layers, num_tokens, num_baseline]
cosines_baseline = (
    torch.einsum("ltd,bd->ltb", hs2d, baseline_embeds_norm).cpu().float().numpy()
)

# Average over baseline dimension (last)
baseline_avg = cosines_baseline.mean(axis=-1)  # [num_layers, num_tokens]

# Calculate difference: target similarity - baseline average
difference = target_cosine_sim - baseline_avg  # [num_layers, num_tokens]

# Prepare tokens for x labels
num_layers, num_tokens = target_cosine_sim.shape
token_ids = input_ids[0, :num_tokens].tolist()
token_strs = tokenizer.convert_ids_to_tokens(token_ids)

# Plot the difference as a heatmap [layers x tokens]
plt.figure(figsize=(min(12, 0.75 * num_tokens), 7))
im = plt.imshow(
    difference,
    aspect="auto",
    cmap="bwr",
    vmin=-np.max(np.abs(difference)),
    vmax=np.max(np.abs(difference)),
)
plt.colorbar(im, label="Target similarity - baseline avg (cosine sim)")
plt.ylabel("Layer")
plt.xlabel("Token Position")
plt.xticks(ticks=np.arange(len(token_strs)), labels=token_strs, rotation=90, fontsize=8)
plt.title(
    f"Difference: '{target_token_str}' embedding vs. avg(1000 random number tokens)"
)
plt.tight_layout()
plt.show()
