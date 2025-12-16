# %%
import torch
from dotenv import load_dotenv

from src.model import CODI

load_dotenv()
# %%
model = CODI.from_pretrained(
    checkpoint_path="bcywinski/codi_llama1b_gsm8k_commonsense-test",  # HF checkpoint ID
    model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",  # HF base model ID
    lora_r=128,
    lora_alpha=32,
    num_latent=6,  # Number of latent reasoning steps
    use_prj=True,  # Whether model uses projection layer
    device="cuda",
    dtype="bfloat16",
    strict=False,
    # Optional: specify where to save the checkpoint (default: ./checkpoints/{name})
    checkpoint_save_path="./checkpoints/bcywinski/codi_llama1b_gsm8k_commonsense-test",
    remove_eos=True,
    full_precision=True,
)
# %%
tokenizer = model.tokenizer
additional_special_tokens = [
    "<|bocot|>",
    "<|eocot|>",
]
tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
tokenizer.bot_id = tokenizer.convert_tokens_to_ids("<|bocot|>")  # beginning of CoT
tokenizer.eot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")
# %%
# Tokenize input
prompt = """Henry and 3 of his friends order 7 pizzas for lunch. Each pizza is cut into 8 slices. If Henry and his friends want to share the pizzas equally, how many slices can each of them have?"""
# prompt = """Question: The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?
# Choices:
# A: ignore
# B: enforce
# C: authoritarian
# D: yell at
# E: avoid"""
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(model.codi.device)
attention_mask = inputs["attention_mask"].to(model.codi.device)
# # %%
# outputs = model.codi.generate(
#     input_ids=input_ids,
#     max_new_tokens=256,
#     temperature=0.1,
#     top_k=40,
#     top_p=0.95,
#     attention_mask=attention_mask,
#     pad_token_id=tokenizer.pad_token_id,
# )
# print(tokenizer.decode(outputs[0], skip_special_tokens=False))
# %%
skip_thinking = False
verbalize_cot = True
# Generate with latent reasoning
output = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    tokenizer=tokenizer,
    max_new_tokens=256,
    num_latent_iterations=6,  # How many latent reasoning steps
    temperature=0.1,
    greedy=True,
    return_latent_vectors=True,  # Get the latent reasoning vectors
    remove_eos=False,
    output_attentions=False,  # Get attention weights
    skip_thinking=skip_thinking,
    output_hidden_states=True,
    verbalize_cot=verbalize_cot,
    sot_token=tokenizer.convert_tokens_to_ids("<|bocot|>"),
    eot_token=tokenizer.convert_tokens_to_ids("<|eocot|>"),
)
# %%
# Decode and print output
generated_text = tokenizer.decode(output["sequences"][0], skip_special_tokens=False)
print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")
# %%
print(f"\nNumber of latent reasoning vectors: {len(output['latent_vectors'])}")
print(f"Each latent vector shape: {output['latent_vectors'][0].shape}")
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
k = 20
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
import matplotlib.pyplot as plt

# Select which latent vector to analyze
selected_latent_idx = 0  # change if you want a different latent vector
selected_latent_vector = latent_vectors[selected_latent_idx]  # shape: [2048]

# Squeeze out singleton batch dimension from hs so shape is [num_layers, num_tokens, hidden_dim]
hs_squeezed = hs.squeeze(1)  # shape: [num_layers, num_tokens, hidden_dim]

# Normalize both hs and selected latent vector for cosine similarity
hs_norm = hs_squeezed / hs_squeezed.norm(
    dim=-1, keepdim=True
)  # (num_layers,num_tokens,hidden_dim)
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
# %%
import matplotlib.pyplot as plt
import numpy as np
import torch

# Choose the specific token (string)
target_token_str = "7"  # Change as needed

# Get token id for the specific token
target_token_id = tokenizer.convert_tokens_to_ids(target_token_str)
if target_token_id is None or target_token_id < 0 or target_token_id >= 128259:
    raise ValueError(
        f"Token '{target_token_str}' not found or out of valid embedding range."
    )

# Get embedding for target token from embedding matrix shape [128259, 2048]
embedding_matrix = (
    model.codi.base_model.model.model.embed_tokens.weight
)  # [128259, 2048]
token_embedding = embedding_matrix[target_token_id].cpu().float()  # [2048]
token_embedding_norm = token_embedding / token_embedding.norm()

# hs_norm shape: [num_layers, batch, num_tokens, hidden_dim]
# Accept batch size of 1 *OR* just pick the first batch (as hs2d confirmed to be [17, 37, 2048])
if hs_norm.ndim == 4:
    hs2d = hs_norm[:, 0, :, :]  # Always use the first batch
else:
    # If already [num_layers, num_tokens, hidden_dim] shape, pass through
    hs2d = hs_norm

hs2d = hs2d.float()
# Compute cosine similarity (dot product) for each layer and token position
cosine_sim = (
    torch.einsum("ltd,d->lt", hs2d, token_embedding_norm).cpu().float().numpy()
)  # [num_layers, num_tokens]

# Normalize similarity to [0, 1]
cosine_sim_min = np.min(cosine_sim)
cosine_sim_max = np.max(cosine_sim)
if cosine_sim_max - cosine_sim_min > 1e-8:
    cosine_sim_norm = (cosine_sim - cosine_sim_min) / (cosine_sim_max - cosine_sim_min)
else:
    cosine_sim_norm = np.zeros_like(cosine_sim)

# Prepare tokens for x labels
num_layers, num_tokens = cosine_sim.shape
token_ids = input_ids[0, :num_tokens].tolist()
token_strs = tokenizer.convert_ids_to_tokens(token_ids)

# Plot the normalized heatmap [layer x token]
plt.figure(figsize=(min(12, 0.75 * num_tokens), 7))
im = plt.imshow(cosine_sim_norm, aspect="auto", cmap="viridis", vmin=0, vmax=1)
plt.colorbar(im, label="Cosine similarity (normalized to [0, 1])")
plt.ylabel("Layer")
plt.xlabel("Token Position")
plt.xticks(ticks=np.arange(len(token_strs)), labels=token_strs, rotation=90, fontsize=8)
plt.title(
    f"Normalized Cosine similarity: '{target_token_str}' embedding vs. activations (layers x tokens)\nEmbed shape: {embedding_matrix.shape}"
)
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt

space_char = "Ġ"
token = "Mexico"
token_id = tokenizer.convert_tokens_to_ids(token)
if token_id is None or token_id < 0:
    print(f"Token '{token}' not found in the tokenizer vocabulary.")
else:
    similarities = [sims[i, token_id].item() for i in range(latent_vectors.shape[0])]
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(similarities) + 1), similarities, marker="o")
    plt.title(
        f"Similarity between each latent vector and token '{token}' (id: {token_id})"
    )
    plt.xlabel("Latent vector #")
    plt.ylabel("Similarity")
    plt.grid(True)
    plt.show()

# %%

token_id

# %%
hs = torch.stack(
    model.codi(
        torch.tensor([token_id]).unsqueeze(0).to(model.codi.device),
        output_hidden_states=True,
    ).hidden_states
).to("cpu")

# %%
hs = hs.squeeze(1).squeeze(1)
hs.shape

# %%
import matplotlib.pyplot as plt

hs = hs.float()
latent_vectors = latent_vectors.float()
# hs: shape [num_layers, hidden_dim] (17, 2048)
# latent_vectors: shape [num_latents, hidden_dim]
num_layers = hs.shape[0]
num_latents = latent_vectors.shape[0]

# Compute cosine similarity between each layer's activation and each latent vector
hs_norm = hs / hs.norm(dim=1, keepdim=True)
lv_norm = latent_vectors / latent_vectors.norm(dim=1, keepdim=True)
sim_matrix = hs_norm @ lv_norm.T  # shape: (17, num_latents)

plt.figure(figsize=(min(2 + 0.6 * num_latents, 14), 6))
im = plt.imshow(sim_matrix, aspect="auto", cmap="viridis")
plt.colorbar(im, label="Cosine similarity")
plt.xlabel("Latent vector")
plt.ylabel("Layer")
plt.title("Cosine similarity between each layer's activation and each latent vector")
plt.xticks(range(num_latents), [f"Latent {i + 1}" for i in range(num_latents)])
plt.yticks(range(num_layers), [f"Layer {i}" for i in range(num_layers)])
plt.tight_layout()
plt.show()

# %%
# Visualize latent token attention patterns
if "attentions" in output and output["attentions"] is not None:
    # Get the generated sequence to identify latent token positions
    generated_ids = output["sequences"][0]
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)

    # Attentions structure: list of tuples (one per generation step)
    # Each tuple contains attention tensors for each layer
    # Shape per layer: (batch_size, num_heads, seq_len, seq_len)

    # Find latent token positions (they should be marked with special token or identifiable)
    # We'll need to track which positions are latent tokens
    # For now, let's visualize attention from the last layer averaged across heads

    print(f"\nNumber of generation steps with attention: {len(output['attentions'])}")

    # Let's focus on the latent tokens during generation
    # We'll collect attention patterns for positions that correspond to latent reasoning

    # For visualization, let's look at attention at specific generation steps
    # where latent tokens are being processed
    layer_idx = -1  # Last layer

    # Collect attention patterns across generation steps
    latent_attention_maps = []
    step_positions = []

    for step_idx, step_attentions in enumerate(output["attentions"]):
        if step_attentions is None or len(step_attentions) == 0:
            continue

        # Get last layer attention for this step
        # Shape: (batch_size, num_heads, current_seq_len, current_seq_len)
        last_layer_attn = step_attentions[layer_idx][0]  # Remove batch dim

        # Average across attention heads
        # Shape: (current_seq_len, current_seq_len)
        avg_attn = last_layer_attn.mean(dim=0).cpu().float()

        # The last position is the newly generated token
        # Get its attention to all previous tokens
        new_token_attn = avg_attn[-1, :]  # Shape: (current_seq_len,)

        latent_attention_maps.append(new_token_attn)
        step_positions.append(step_idx)

    if latent_attention_maps:
        # Find the maximum sequence length to pad all attention vectors
        max_len = max(attn.shape[0] for attn in latent_attention_maps)

        # Pad attention vectors and stack them
        padded_attns = []
        for attn in latent_attention_maps:
            if attn.shape[0] < max_len:
                padding = torch.zeros(max_len - attn.shape[0])
                attn = torch.cat([attn, padding])
            padded_attns.append(attn)

        attention_matrix = torch.stack(padded_attns).numpy()

        # Create heatmap
        plt.figure(figsize=(16, max(8, len(latent_attention_maps) * 0.3)))
        im = plt.imshow(
            attention_matrix, aspect="auto", cmap="viridis", interpolation="nearest"
        )
        plt.colorbar(im, label="Attention weight")

        # Set labels
        plt.xlabel("Token position in sequence")
        plt.ylabel("Generation step")
        plt.title(
            "Attention patterns during generation (last layer, averaged across heads)"
        )

        # Add token labels on x-axis for reference (sample every N tokens if too many)
        num_positions = attention_matrix.shape[1]
        if num_positions <= 50:
            step = 1
        elif num_positions <= 100:
            step = 2
        else:
            step = num_positions // 50

        xtick_positions = list(range(0, num_positions, step))
        xtick_labels = []
        for pos in xtick_positions:
            if pos < len(generated_tokens):
                token = generated_tokens[pos]
                # Truncate long tokens
                if len(token) > 10:
                    token = token[:10] + "..."
                xtick_labels.append(f"{pos}:{token}")
            else:
                xtick_labels.append(f"{pos}")

        plt.xticks(xtick_positions, xtick_labels, rotation=90, ha="right", fontsize=8)
        plt.yticks(range(len(step_positions)), [f"Step {i}" for i in step_positions])

        plt.tight_layout()
        plt.savefig("latent_token_attention.png", dpi=150, bbox_inches="tight")
        plt.show()

        print("\nAttention heatmap saved to 'latent_token_attention.png'")
        print(
            f"Heatmap shape: {attention_matrix.shape} (generation_steps, sequence_length)"
        )

        # Also create a focused view on latent iterations if we can identify them
        # Let's look at specific steps that correspond to latent reasoning
        num_latent = model.num_latent
        num_latent_iterations = 2  # From the generate call

        print(f"\nModel has {num_latent} latent tokens per iteration")
        print(f"Using {num_latent_iterations} latent iterations")

else:
    print(
        "\nNo attention weights found in output. The model may not support returning attentions."
    )

# %%
prompt = """Henry and 3 of his friends order 8 pizzas for lunch. Each pizza is cut into 8 slices. If Henry and his friends want to share the pizzas equally, how many slices can each of them have?"""
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
input_ids = inputs["input_ids"].to(model.codi.device)
attention_mask = inputs["attention_mask"].to(model.codi.device)
# %%
print(tokenizer.convert_ids_to_tokens(input_ids[0]))
# %%
# Generate with latent reasoning
output = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    tokenizer=tokenizer,
    max_new_tokens=256,
    num_latent_iterations=6,  # How many latent reasoning steps
    temperature=0.1,
    greedy=True,
    return_latent_vectors=True,  # Get the latent reasoning vectors
    remove_eos=True,
    output_attentions=False,  # Get attention weights
    first_latent_emb=latent_vectors[0].unsqueeze(0).to(model.codi.device),
)

# %%
# Decode and print output
generated_text = tokenizer.decode(output["sequences"][0], skip_special_tokens=False)
print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")
print(f"\nNumber of latent reasoning vectors: {len(output['latent_vectors'])}")
print(f"Each latent vector shape: {output['latent_vectors'][0].shape}")
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
k = 20
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
