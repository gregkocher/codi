# ABOUTME: Attention visualization for CODI latent vectors.
# ABOUTME: Shows 2D heatmaps of how each token/latent position attends to other positions.

# %%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import CODI

# %%
# Parameters
CHECKPOINT_PATH = "bcywinski/codi_llama1b-answer_only"
MODEL_NAME_OR_PATH = "meta-llama/Llama-3.2-1B-Instruct"

DEVICE = "cuda"
DTYPE = "bfloat16"

PROMPT = "A team starts with 3 members. They recruit 5 new members. Then each current member recruits 2 additional people. How many people are there now on the team? Give the answer only and nothing else."
NUM_LATENT_ITERATIONS = 6
MAX_NEW_TOKENS = 32

# Visualization options
LAYER_TO_VISUALIZE = -1  # -1 for last layer, or specific layer index
HEAD_TO_VISUALIZE = None  # None for average across heads, or specific head index
SHOW_LAST_N_PROMPT_TOKENS = 1000  # Set high to show all prompt tokens


# %%
def run_inference_with_attention(
    model, tokenizer, prompt, num_latent_iterations, max_new_tokens=64, greedy=True
):
    """
    Run CODI inference and capture attention weights for each position.

    Returns:
        Dict with attention patterns for prompt, latent iterations, and answer tokens
    """
    device = next(model.parameters()).device

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Get token strings for labeling
    prompt_tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    # Special tokens
    sot_token = tokenizer.convert_tokens_to_ids("<|bocot|>")
    eot_token = tokenizer.convert_tokens_to_ids("<|eocot|>")

    # Add BOT tokens
    bot_tensor = torch.tensor(
        [tokenizer.eos_token_id, sot_token], dtype=torch.long, device=device
    ).unsqueeze(0)

    input_ids_bot = torch.cat((input_ids, bot_tensor), dim=1)
    attention_mask_bot = torch.cat((attention_mask, torch.ones_like(bot_tensor)), dim=1)

    # Update token list
    prompt_tokens.append(tokenizer.decode([tokenizer.eos_token_id]))
    prompt_tokens.append("<|bocot|>")

    results = {
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "num_latent_iterations": num_latent_iterations,
        "prompt_attention": None,  # Attention during prompt encoding
        "latent_attentions": [],  # Attention for each latent iteration
        "answer_attentions": [],  # Attention for each answer token
        "answer_tokens": [],  # Generated answer tokens
    }

    with torch.no_grad():
        # Encode prompt with attention
        outputs = model.codi(
            input_ids=input_ids_bot,
            use_cache=True,
            output_hidden_states=True,
            output_attentions=True,
            attention_mask=attention_mask_bot,
        )
        past_key_values = outputs.past_key_values

        # Store prompt attention (full seq x seq matrix)
        # attentions is tuple of (batch, heads, seq, seq) per layer
        prompt_attn = torch.stack(
            [a.squeeze(0) for a in outputs.attentions], dim=0
        )  # (layers, heads, seq, seq)
        results["prompt_attention"] = prompt_attn.cpu()

        # Get initial latent embedding
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        if model.use_prj:
            latent_embd = model.prj(latent_embd)
            latent_embd = latent_embd.to(dtype=model.codi.dtype)

        # Latent iterations
        current_seq_len = input_ids_bot.size(1)

        for i in range(num_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values

            # Store latent attention
            # Each attention is (batch, heads, 1, current_total_seq_len)
            latent_attn = torch.stack(
                [a.squeeze(0) for a in outputs.attentions], dim=0
            )  # (layers, heads, 1, seq)
            results["latent_attentions"].append(
                {
                    "iteration": i,
                    "attention": latent_attn.cpu(),
                    "seq_len": current_seq_len + 1,  # Including this latent
                }
            )

            current_seq_len += 1

            # Get next latent embedding
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                latent_embd = model.prj(latent_embd)
                latent_embd = latent_embd.to(dtype=model.codi.dtype)

        # Add EOT token and generate answer
        eot_ids = torch.tensor([[eot_token]], dtype=torch.long, device=device)
        eot_emb = model.get_embd(model.codi, model.model_name)(eot_ids)

        outputs = model.codi(
            inputs_embeds=eot_emb,
            use_cache=True,
            output_hidden_states=True,
            output_attentions=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values
        current_seq_len += 1

        # Store EOT attention
        eot_attn = torch.stack(
            [a.squeeze(0) for a in outputs.attentions], dim=0
        )
        results["answer_attentions"].append(
            {
                "token": "<|eocot|>",
                "attention": eot_attn.cpu(),
                "seq_len": current_seq_len,
            }
        )
        results["answer_tokens"].append("<|eocot|>")

        # Generate answer tokens
        logits = outputs.logits[:, -1, :]

        for step in range(max_new_tokens):
            if greedy:
                next_token_id = torch.argmax(logits, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Check for EOS
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            # Get token string
            token_str = tokenizer.decode([next_token_id.item()])
            results["answer_tokens"].append(token_str)

            # Get embedding for next token
            next_emb = model.get_embd(model.codi, model.model_name)(
                next_token_id.unsqueeze(0)
            )

            # Forward pass with attention
            outputs = model.codi(
                inputs_embeds=next_emb,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            current_seq_len += 1

            # Store answer token attention
            ans_attn = torch.stack(
                [a.squeeze(0) for a in outputs.attentions], dim=0
            )
            results["answer_attentions"].append(
                {
                    "token": token_str,
                    "attention": ans_attn.cpu(),
                    "seq_len": current_seq_len,
                }
            )

            logits = outputs.logits[:, -1, :]

    print(f"Generated answer: {''.join(results['answer_tokens'])}")
    return results


# %%
def visualize_latent_attention_to_all(
    results,
    layer_idx=-1,
    head_idx=None,
    last_n_prompt_tokens=20,
    figsize=(14, 8),
):
    """
    Visualize how each latent position attends to all previous positions.

    Creates a 2D heatmap where:
    - Rows: latent iterations (L0, L1, L2, ...)
    - Columns: all positions (last N prompt tokens + previous latents)
    """
    prompt_tokens = results["prompt_tokens"]
    num_latent = len(results["latent_attentions"])

    if num_latent == 0:
        print("No latent positions to visualize")
        return None

    # Get number of layers from first latent attention
    num_layers = results["latent_attentions"][0]["attention"].shape[0]
    if layer_idx < 0:
        layer_idx = num_layers + layer_idx

    # Determine which prompt tokens to show (skip first token)
    num_prompt_tokens = len(prompt_tokens)
    start_prompt_idx = max(1, num_prompt_tokens - last_n_prompt_tokens)
    shown_prompt_tokens = prompt_tokens[start_prompt_idx:]

    # Build attention matrix
    # Each row is a latent position
    # Columns are: [shown_prompt_tokens] + [L0, L1, ..., L_{i-1}]
    # But each latent sees different number of previous latents

    # For simplicity, show attention to:
    # - Last N prompt tokens
    # - All latent positions (padded with 0 for future latents)
    num_cols = len(shown_prompt_tokens) + num_latent
    attn_matrix = np.zeros((num_latent, num_cols))

    for i, latent_data in enumerate(results["latent_attentions"]):
        attn = latent_data["attention"]  # (layers, heads, 1, seq_len)

        # Select layer
        layer_attn = attn[layer_idx]  # (heads, 1, seq_len)

        # Average or select head
        if head_idx is None:
            head_attn = layer_attn.mean(dim=0)  # (1, seq_len)
        else:
            head_attn = layer_attn[head_idx]  # (1, seq_len)

        head_attn = head_attn.squeeze(0).float().numpy()  # (seq_len,)

        # Extract attention to shown prompt tokens
        prompt_attn = head_attn[start_prompt_idx:num_prompt_tokens]
        attn_matrix[i, : len(prompt_attn)] = prompt_attn

        # Extract attention to previous latents (if any)
        if i > 0:
            latent_attn = head_attn[num_prompt_tokens : num_prompt_tokens + i]
            attn_matrix[i, len(shown_prompt_tokens) : len(shown_prompt_tokens) + i] = (
                latent_attn
            )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(attn_matrix, cmap="Blues", aspect="auto", vmin=0)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attention weight", rotation=270, labelpad=15)

    # X-axis labels
    x_labels = [repr(t)[1:-1][:8] for t in shown_prompt_tokens] + [
        f"L{i}" for i in range(num_latent)
    ]
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)

    # Add vertical line to separate prompt from latents
    ax.axvline(x=len(shown_prompt_tokens) - 0.5, color="red", linestyle="--", alpha=0.7)

    # Y-axis labels
    ax.set_yticks(range(num_latent))
    ax.set_yticklabels([f"L{i}" for i in range(num_latent)])

    ax.set_xlabel("Attended position (prompt tokens | latent positions)")
    ax.set_ylabel("Query position (latent iteration)")

    head_str = f"head {head_idx}" if head_idx is not None else "avg heads"
    ax.set_title(f"Latent Attention Pattern (layer {layer_idx}, {head_str})")

    plt.tight_layout()
    return fig


# %%
def visualize_single_latent_attention(
    results,
    latent_idx=0,
    layer_idx=-1,
    head_idx=None,
    last_n_prompt_tokens=30,
    figsize=(12, 4),
):
    """
    Visualize attention for a single latent position as a bar chart.
    """
    prompt_tokens = results["prompt_tokens"]
    num_prompt_tokens = len(prompt_tokens)

    if latent_idx >= len(results["latent_attentions"]):
        print(f"Latent index {latent_idx} out of range")
        return None

    latent_data = results["latent_attentions"][latent_idx]
    attn = latent_data["attention"]  # (layers, heads, 1, seq_len)

    num_layers = attn.shape[0]
    if layer_idx < 0:
        layer_idx = num_layers + layer_idx

    layer_attn = attn[layer_idx]  # (heads, 1, seq_len)

    if head_idx is None:
        head_attn = layer_attn.mean(dim=0).squeeze(0).float().numpy()
    else:
        head_attn = layer_attn[head_idx].squeeze(0).float().numpy()

    # Determine which tokens to show (skip first token)
    start_idx = max(1, num_prompt_tokens - last_n_prompt_tokens)

    # Build labels and values
    labels = []
    values = []

    # Prompt tokens
    for i in range(start_idx, num_prompt_tokens):
        token = prompt_tokens[i]
        labels.append(repr(token)[1:-1][:10])
        values.append(head_attn[i])

    # Previous latent positions
    for i in range(latent_idx):
        labels.append(f"L{i}")
        values.append(head_attn[num_prompt_tokens + i])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    colors = ["steelblue"] * (num_prompt_tokens - start_idx) + ["coral"] * latent_idx
    ax.bar(range(len(values)), values, color=colors)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    # Add vertical line
    prompt_end = num_prompt_tokens - start_idx
    if latent_idx > 0:
        ax.axvline(x=prompt_end - 0.5, color="red", linestyle="--", alpha=0.7)

    ax.set_xlabel("Attended position")
    ax.set_ylabel("Attention weight")

    head_str = f"head {head_idx}" if head_idx is not None else "avg heads"
    ax.set_title(f"Attention from L{latent_idx} (layer {layer_idx}, {head_str})")

    plt.tight_layout()
    return fig


# %%
def visualize_attention_across_layers(
    results,
    latent_idx=0,
    head_idx=None,
    last_n_prompt_tokens=20,
    figsize=(14, 10),
):
    """
    Visualize attention across all layers for a single latent position.

    Creates a 2D heatmap where:
    - Rows: layers
    - Columns: attended positions
    """
    prompt_tokens = results["prompt_tokens"]
    num_prompt_tokens = len(prompt_tokens)

    if latent_idx >= len(results["latent_attentions"]):
        print(f"Latent index {latent_idx} out of range")
        return None

    latent_data = results["latent_attentions"][latent_idx]
    attn = latent_data["attention"]  # (layers, heads, 1, seq_len)

    num_layers = attn.shape[0]

    # Average or select head
    if head_idx is None:
        layer_attn = attn.mean(dim=1).squeeze(1).float().numpy()  # (layers, seq_len)
    else:
        layer_attn = attn[:, head_idx].squeeze(1).float().numpy()  # (layers, seq_len)

    # Determine which tokens to show (skip first token)
    start_idx = max(1, num_prompt_tokens - last_n_prompt_tokens)

    # Build attention matrix
    num_cols = (num_prompt_tokens - start_idx) + latent_idx
    attn_matrix = np.zeros((num_layers, num_cols))

    for layer in range(num_layers):
        # Prompt tokens
        attn_matrix[layer, : (num_prompt_tokens - start_idx)] = layer_attn[
            layer, start_idx:num_prompt_tokens
        ]
        # Previous latents
        if latent_idx > 0:
            attn_matrix[layer, (num_prompt_tokens - start_idx) :] = layer_attn[
                layer, num_prompt_tokens : num_prompt_tokens + latent_idx
            ]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(attn_matrix, cmap="Blues", aspect="auto", vmin=0)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attention weight", rotation=270, labelpad=15)

    # X-axis labels
    x_labels = [repr(t)[1:-1][:8] for t in prompt_tokens[start_idx:]] + [
        f"L{i}" for i in range(latent_idx)
    ]
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)

    # Vertical line
    if latent_idx > 0:
        ax.axvline(
            x=(num_prompt_tokens - start_idx) - 0.5,
            color="red",
            linestyle="--",
            alpha=0.7,
        )

    # Y-axis labels
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f"{i}" for i in range(num_layers)])

    ax.set_xlabel("Attended position")
    ax.set_ylabel("Layer")

    head_str = f"head {head_idx}" if head_idx is not None else "avg heads"
    ax.set_title(f"Attention from L{latent_idx} across layers ({head_str})")

    plt.tight_layout()
    return fig


# %%
def visualize_attention_avg_all(
    results,
    last_n_prompt_tokens=1000,
    figsize=(14, 10),
):
    """
    Visualize attention averaged over both heads and layers.

    Creates a 2D heatmap where:
    - Rows: latent iterations (L0, L1, ...) + answer tokens (A0, A1, ...)
    - Columns: all positions (prompt tokens + latents + previous answer tokens)
    """
    prompt_tokens = results["prompt_tokens"]
    num_latent = len(results["latent_attentions"])
    num_answer = len(results.get("answer_attentions", []))
    answer_tokens = results.get("answer_tokens", [])

    if num_latent == 0:
        print("No latent positions to visualize")
        return None

    # Determine which prompt tokens to show (skip first token)
    num_prompt_tokens = len(prompt_tokens)
    start_prompt_idx = max(1, num_prompt_tokens - last_n_prompt_tokens)
    shown_prompt_tokens = prompt_tokens[start_prompt_idx:]

    # Build attention matrix
    # Columns: shown_prompt_tokens + latents + answer tokens
    num_cols = len(shown_prompt_tokens) + num_latent + num_answer
    num_rows = num_latent + num_answer
    attn_matrix = np.zeros((num_rows, num_cols))

    # Fill in latent attention rows
    for i, latent_data in enumerate(results["latent_attentions"]):
        attn = latent_data["attention"]  # (layers, heads, 1, seq_len)

        # Average over both layers and heads
        avg_attn = attn.mean(dim=0).mean(dim=0).squeeze(0).float().numpy()  # (seq_len,)

        # Extract attention to shown prompt tokens
        prompt_attn = avg_attn[start_prompt_idx:num_prompt_tokens]
        attn_matrix[i, : len(prompt_attn)] = prompt_attn

        # Extract attention to previous latents (if any)
        if i > 0:
            latent_attn = avg_attn[num_prompt_tokens : num_prompt_tokens + i]
            attn_matrix[i, len(shown_prompt_tokens) : len(shown_prompt_tokens) + i] = (
                latent_attn
            )

    # Fill in answer attention rows
    for j, ans_data in enumerate(results.get("answer_attentions", [])):
        row_idx = num_latent + j
        attn = ans_data["attention"]  # (layers, heads, 1, seq_len)

        # Average over both layers and heads
        avg_attn = attn.mean(dim=0).mean(dim=0).squeeze(0).float().numpy()  # (seq_len,)

        # Extract attention to shown prompt tokens
        prompt_attn = avg_attn[start_prompt_idx:num_prompt_tokens]
        attn_matrix[row_idx, : len(prompt_attn)] = prompt_attn

        # Extract attention to latents
        latent_attn = avg_attn[num_prompt_tokens : num_prompt_tokens + num_latent]
        attn_matrix[
            row_idx, len(shown_prompt_tokens) : len(shown_prompt_tokens) + num_latent
        ] = latent_attn

        # Extract attention to previous answer tokens (if any)
        if j > 0:
            ans_attn = avg_attn[
                num_prompt_tokens + num_latent : num_prompt_tokens + num_latent + j
            ]
            attn_matrix[
                row_idx,
                len(shown_prompt_tokens) + num_latent : len(shown_prompt_tokens)
                + num_latent
                + j,
            ] = ans_attn

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(attn_matrix, cmap="Blues", aspect="auto", vmin=0)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attention weight", rotation=270, labelpad=15)

    # X-axis labels
    x_labels = (
        [repr(t)[1:-1][:8] for t in shown_prompt_tokens]
        + [f"L{i}" for i in range(num_latent)]
        + [repr(t)[1:-1][:6] for t in answer_tokens]
    )
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)

    # Add vertical lines to separate sections
    ax.axvline(x=len(shown_prompt_tokens) - 0.5, color="red", linestyle="--", alpha=0.7)
    if num_answer > 0:
        ax.axvline(
            x=len(shown_prompt_tokens) + num_latent - 0.5,
            color="green",
            linestyle="--",
            alpha=0.7,
        )

    # Add horizontal line to separate latents from answer tokens
    if num_answer > 0:
        ax.axhline(y=num_latent - 0.5, color="green", linestyle="--", alpha=0.7)

    # Y-axis labels
    y_labels = [f"L{i}" for i in range(num_latent)] + [
        repr(t)[1:-1][:6] for t in answer_tokens
    ]
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)

    ax.set_xlabel("Attended position (prompt | latents | answer)")
    ax.set_ylabel("Query position (latents | answer)")
    ax.set_title("Attention Pattern (averaged over all layers and heads)")

    plt.tight_layout()
    return fig


# %%
def ensure_tokenizer_special_tokens(tokenizer, model) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = model.pad_token_id
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|bocot|>", "<|eocot|>"]}
    )


# %%
load_dotenv()

print("Loading model...")
model = CODI.from_pretrained(
    checkpoint_path=CHECKPOINT_PATH,
    model_name_or_path=MODEL_NAME_OR_PATH,
    lora_r=128,
    lora_alpha=32,
    num_latent=6,
    use_prj=True,
    device=DEVICE,
    dtype=DTYPE,
    strict=False,
    checkpoint_save_path=f"./checkpoints/{CHECKPOINT_PATH}",
    remove_eos=True,
    full_precision=True,
)
tokenizer = model.tokenizer
ensure_tokenizer_special_tokens(tokenizer, model)

# %%
print("Running inference with attention capture...")
results = run_inference_with_attention(
    model=model,
    tokenizer=tokenizer,
    prompt=PROMPT,
    num_latent_iterations=NUM_LATENT_ITERATIONS,
    max_new_tokens=MAX_NEW_TOKENS,
)

# %%
# Visualization 0: All latents attending to all positions (averaged over all layers and heads)
print("\n0. Latent attention pattern (averaged over all layers and heads)...")
fig0 = visualize_attention_avg_all(
    results,
    last_n_prompt_tokens=SHOW_LAST_N_PROMPT_TOKENS,
)
if fig0 is not None:
    output_path = "/workspace/projects/codi/outputs/attention_avg_all.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig0.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()

# %%
# Visualization 1: All latents attending to all positions (single layer)
print("\n1. Latent attention pattern (all latents to all positions)...")
fig1 = visualize_latent_attention_to_all(
    results,
    layer_idx=LAYER_TO_VISUALIZE,
    head_idx=HEAD_TO_VISUALIZE,
    last_n_prompt_tokens=SHOW_LAST_N_PROMPT_TOKENS,
)
if fig1 is not None:
    output_path = "/workspace/projects/codi/outputs/attention_latents_to_all.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig1.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()

# %%
# Visualization 2: Single latent attention as bar chart
print("\n2. Single latent attention (bar chart)...")
fig2 = visualize_single_latent_attention(
    results,
    latent_idx=NUM_LATENT_ITERATIONS - 1,  # Last latent
    layer_idx=LAYER_TO_VISUALIZE,
    head_idx=HEAD_TO_VISUALIZE,
    last_n_prompt_tokens=30,
)
if fig2 is not None:
    output_path = "/workspace/projects/codi/outputs/attention_single_latent.png"
    fig2.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()

# %%
# Visualization 3: Attention across layers for single latent
print("\n3. Attention across layers...")
fig3 = visualize_attention_across_layers(
    results,
    latent_idx=NUM_LATENT_ITERATIONS - 1,  # Last latent
    head_idx=HEAD_TO_VISUALIZE,
    last_n_prompt_tokens=SHOW_LAST_N_PROMPT_TOKENS,
)
if fig3 is not None:
    output_path = "/workspace/projects/codi/outputs/attention_across_layers.png"
    fig3.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()

# %%
# Print some statistics
print("\n" + "=" * 60)
print("Attention Statistics")
print("=" * 60)

num_prompt = len(results["prompt_tokens"])
for i, latent_data in enumerate(results["latent_attentions"]):
    attn = latent_data["attention"]  # (layers, heads, 1, seq_len)
    last_layer_attn = attn[-1].mean(dim=0).squeeze(0).float().numpy()  # (seq_len,)

    # Attention to prompt vs latents
    prompt_attn_sum = last_layer_attn[:num_prompt].sum()
    latent_attn_sum = last_layer_attn[num_prompt:].sum() if i > 0 else 0

    # Top attended positions
    top_indices = np.argsort(last_layer_attn)[-5:][::-1]
    top_tokens = []
    for idx in top_indices:
        if idx < num_prompt:
            top_tokens.append(f"{repr(results['prompt_tokens'][idx])[:10]}")
        else:
            top_tokens.append(f"L{idx - num_prompt}")

    print(f"L{i}: prompt_attn={prompt_attn_sum:.3f}, latent_attn={latent_attn_sum:.3f}")
    print(f"    Top attended: {', '.join(top_tokens)}")

# %%
