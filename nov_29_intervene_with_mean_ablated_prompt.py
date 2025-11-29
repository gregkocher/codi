# %%
"""
Mean Ablation Experiment: Testing the effect of mean-ablating prompt activations
on model performance with different latent reasoning strategies.

This script compares four conditions:
1. Baseline: Normal generation
2. Mean-ablated + Frozen Latents: Mean-ablated prompt with all latent activations frozen
3. Mean-ablated + Regenerated Latents: Mean-ablated prompt with latents regenerated normally
4. Mean-ablated + Patched Embeddings: Mean-ablated prompt with only latent embeddings patched
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from src.datasets import extract_answer_number
from src.model import CODI

# ============================================================================
# Utility Functions
# ============================================================================


def load_latent_vectors(
    metadata_file="nov_27_all_test_cases_metadata.json",
    latent_vectors_file="nov_27_all_latent_vectors.npy",
    valid_mask_file="nov_27_valid_mask.npy",
):
    """Load metadata and latent vectors from files."""
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    latent_vectors = np.load(latent_vectors_file)
    valid_mask = np.load(valid_mask_file)
    return metadata, latent_vectors, valid_mask


def get_average_latent_vector(
    metadata, latent_vectors, valid_mask, latent_idx, filter_dict=None
):
    """Get average latent vector for a specific latent index, optionally filtered."""
    indices = np.where(valid_mask)[0]

    if filter_dict:
        filtered_indices = []
        for idx in indices:
            match = True
            for key, value in filter_dict.items():
                if metadata[idx].get(key) != value:
                    match = False
                    break
            if match:
                filtered_indices.append(idx)
        indices = np.array(filtered_indices)

    if len(indices) == 0:
        print(f"No cases match the filter criteria: {filter_dict}")
        return None

    selected_latents = latent_vectors[indices, latent_idx, :]
    avg_vector = np.mean(selected_latents, axis=0)

    print(f"Computed average for latent {latent_idx} across {len(indices)} cases")
    if filter_dict:
        print(f"Filter: {filter_dict}")

    return avg_vector


def get_transformer_layers(model):
    """Robustly retrieve the ModuleList of decoder layers, handling PEFT wrappers."""
    obj = model.codi

    if hasattr(obj, "get_base_model"):
        obj = obj.get_base_model()

    if hasattr(obj, "model"):
        obj = obj.model

    if hasattr(obj, "model"):
        obj = obj.model

    if hasattr(obj, "layers"):
        return obj.layers

    # Fallback search
    print("Warning: Standard layer path not found. Searching modules...")
    for name, module in model.codi.named_modules():
        if "layers" in name and isinstance(module, torch.nn.ModuleList):
            return module

    raise AttributeError(f"Could not find transformer layers in {type(model.codi)}")


def prepare_inputs(model, tokenizer, prompt):
    """Construct input sequence: [Prompt Tokens] + [BOCOT]"""
    device = model.codi.device

    inputs = tokenizer(
        prompt, return_tensors="pt", padding=False, add_special_tokens=True
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Add BOCOT token
    bot_id = tokenizer.convert_tokens_to_ids("<|bocot|>")
    input_ids_bot = torch.cat(
        [input_ids, torch.tensor([[bot_id]], device=device)], dim=1
    )
    attention_mask_bot = torch.cat(
        [attention_mask, torch.ones((1, 1), device=device)], dim=1
    )

    return input_ids_bot, attention_mask_bot


# ============================================================================
# Activation Capture Functions
# ============================================================================


def capture_prompt_activations(model, tokenizer, prompt):
    """Capture residual stream activations for all layers and all prompt tokens."""
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)
    device = model.codi.device
    layers = get_transformer_layers(model)
    num_layers = len(layers)

    captured_prefill = [None] * num_layers
    handles = []

    def get_prefill_hook(layer_idx):
        def hook(module, args, output):
            act = output[0] if isinstance(output, tuple) else output
            captured_prefill[layer_idx] = act.detach().cpu()

        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(get_prefill_hook(i)))

    with torch.no_grad():
        outputs = model.codi(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )

    for h in handles:
        h.remove()

    # Stack activations: (NumLayers, PromptSeqLen, HiddenDim)
    seq_len = input_ids.shape[1] - 1  # Exclude BOCOT token
    prefill_tensor = torch.stack([x[0, :seq_len, :] for x in captured_prefill])

    return {"prefill": prefill_tensor, "input_ids": input_ids[:, :seq_len]}


def compute_mean_prompt_activations(model, tokenizer, template, combinations):
    """Compute mean activations across multiple template prompts using specific combinations."""
    print(f"Computing mean prompt activations across {len(combinations)} samples...")

    # Generate prompts from combinations
    prompts = []
    for X, Y, Z in combinations:
        prompt = template.format(X=X, Y=Y, Z=Z)
        prompts.append(prompt)

    all_activations = []
    all_seq_lens = []

    for prompt in tqdm(prompts, desc="Capturing activations"):
        result = capture_prompt_activations(model, tokenizer, prompt)
        activations = result["prefill"]
        seq_len = activations.shape[1]
        all_activations.append(activations)
        all_seq_lens.append(seq_len)

    max_seq_len = max(all_seq_lens)
    num_layers = all_activations[0].shape[0]
    hidden_dim = all_activations[0].shape[2]

    # Compute mean activations position by position
    mean_activations_list = []
    for pos in range(max_seq_len):
        pos_activations = []
        for act in all_activations:
            seq_len = act.shape[1]
            if pos < seq_len:
                pos_activations.append(act[:, pos, :])

        if len(pos_activations) > 0:
            stacked_pos = torch.stack(pos_activations, dim=0)
            mean_pos = torch.mean(stacked_pos, dim=0)
            mean_activations_list.append(mean_pos)
        else:
            mean_activations_list.append(torch.zeros(num_layers, hidden_dim))

    mean_activations = torch.stack(mean_activations_list, dim=1)

    return {
        "mean_prefill": mean_activations,
        "token_counts": torch.tensor(all_seq_lens),
        "max_seq_len": max_seq_len,
    }


def capture_target_frozen_latents(model, tokenizer, prompt, num_latents=6):
    """Capture latent activations from a prompt run for freezing."""
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)
    device = model.codi.device
    layers = get_transformer_layers(model)
    num_layers = len(layers)

    captured_latents = [[] for _ in range(num_layers)]
    handles = []

    def get_latent_hook(layer_idx):
        def hook(module, args, output):
            act = output[0] if isinstance(output, tuple) else output
            captured_latents[layer_idx].append(act.detach().cpu())

        return hook

    with torch.no_grad():
        # Prefill phase
        outputs = model.codi(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )

        past_key_values_after_prefill = outputs.past_key_values
        initial_latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        if model.use_prj:
            initial_latent_embd = model.prj(initial_latent_embd).to(
                dtype=model.codi.dtype
            )

        # Latent loop - capture all activations
        for i, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(get_latent_hook(i)))

        past_key_values_list = [past_key_values_after_prefill]
        latent_embd = initial_latent_embd

        for i in range(num_latents):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                past_key_values=past_key_values_list[-1],
                output_hidden_states=True,
            )
            past_key_values_list.append(outputs.past_key_values)
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        for h in handles:
            h.remove()

    # Consolidate: (Layers, NumLatents, Dim)
    latent_stacked_layers = []
    for layer_list in captured_latents:
        if len(layer_list) > 0:
            layer_steps = torch.cat(layer_list, dim=0)
            latent_stacked_layers.append(layer_steps)
        else:
            hidden_dim = initial_latent_embd.shape[-1]
            latent_stacked_layers.append(torch.zeros(num_latents, hidden_dim))
    latent_tensor = torch.stack(latent_stacked_layers)

    return {
        "past_key_values_after_prefill": past_key_values_after_prefill,
        "initial_latent_embd": initial_latent_embd,
        "latent_activations": latent_tensor,
        "past_key_values_list": past_key_values_list,
    }


def capture_latent_embeddings(model, tokenizer, prompt, num_latents=6):
    """Capture the latent embeddings (input tokens) used during latent thinking."""
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)
    device = model.codi.device

    with torch.no_grad():
        outputs = model.codi(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )

        past_key_values = outputs.past_key_values
        initial_latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        if model.use_prj:
            initial_latent_embd = model.prj(initial_latent_embd).to(
                dtype=model.codi.dtype
            )

        latent_embeddings = [initial_latent_embd.clone()]
        latent_embd = initial_latent_embd

        for i in range(num_latents):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

            if i < num_latents - 1:
                latent_embeddings.append(latent_embd.clone())

    return {"latent_embeddings": latent_embeddings}


def compute_mean_latent_embeddings_across_other_templates(
    model, tokenizer, current_template_idx, X, Y, Z, num_latents=6
):
    """
    Compute mean latent embeddings by averaging across all other templates
    (excluding current_template_idx) using the same X, Y, Z values.
    """
    num_templates = get_num_templates()
    all_latent_embeddings = []

    for template_idx in range(num_templates):
        if template_idx == current_template_idx:
            continue

        template_str, param_mapping = get_template_and_param_mapping(template_idx)
        mapped_X, mapped_Y, mapped_Z = param_mapping(X, Y, Z)
        prompt = template_str.format(X=mapped_X, Y=mapped_Y, Z=mapped_Z)

        latent_embeddings_dict = capture_latent_embeddings(
            model, tokenizer, prompt, num_latents=num_latents
        )
        all_latent_embeddings.append(latent_embeddings_dict["latent_embeddings"])

    # Average latent embeddings across all other templates
    num_other_templates = len(all_latent_embeddings)
    if num_other_templates == 0:
        raise ValueError("No other templates to average over")

    # Stack embeddings: (num_other_templates, num_latents, batch_size, seq_len, hidden_dim)
    num_latent_steps = len(all_latent_embeddings[0])
    mean_latent_embeddings = []

    for step_idx in range(num_latent_steps):
        step_embeddings = [emb[step_idx] for emb in all_latent_embeddings]
        stacked_step = torch.stack(step_embeddings, dim=0)
        mean_step = torch.mean(stacked_step, dim=0)
        mean_latent_embeddings.append(mean_step)

    return {"latent_embeddings": mean_latent_embeddings}


# ============================================================================
# Generation Functions with Interventions
# ============================================================================


def _create_mean_ablation_hook(
    layer_idx, mean_prefill, prompt_seq_len, max_seq_len, device
):
    """Create a hook function that replaces prompt activations with mean activations."""

    def hook(module, args, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            new_hidden = hidden_states.clone()
            seq_len = min(hidden_states.shape[1], prompt_seq_len, max_seq_len)
            for token_idx in range(seq_len):
                new_hidden[:, token_idx, :] = (
                    mean_prefill[layer_idx, token_idx, :].unsqueeze(0).to(device)
                )
            return (new_hidden,) + output[1:]
        else:
            new_hidden = output.clone()
            seq_len = min(output.shape[1], prompt_seq_len, max_seq_len)
            for token_idx in range(seq_len):
                new_hidden[:, token_idx, :] = (
                    mean_prefill[layer_idx, token_idx, :].unsqueeze(0).to(device)
                )
            return new_hidden

    return hook


def generate_with_mean_ablated_prompt_frozen_latents(
    model,
    tokenizer,
    prompt,
    mean_activations_dict,
    frozen_latents_dict,
    max_new_tokens=128,
    temperature=0.1,
    greedy=True,
):
    """
    Condition 2: Mean-ablated prompt + frozen latents.
    Mean-ablates prompt activations, then freezes all latent layer activations.
    """
    device = model.codi.device
    layers = get_transformer_layers(model)
    num_layers = len(layers)

    mean_prefill = mean_activations_dict["mean_prefill"].to(device)
    max_seq_len = mean_prefill.shape[1]
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)
    prompt_seq_len = input_ids.shape[1] - 1

    # Set up mean ablation hooks for prefill
    patch_handles = []
    for layer_idx, layer in enumerate(layers):
        hook = _create_mean_ablation_hook(
            layer_idx, mean_prefill, prompt_seq_len, max_seq_len, device
        )
        patch_handles.append(layer.register_forward_hook(hook))

    frozen_latents = frozen_latents_dict["latent_activations"].to(device)
    num_latents = frozen_latents.shape[1]
    frozen_latent_handles = []

    def get_frozen_latent_hook(latent_step, layer_idx):
        def hook(module, args, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                new_hidden = hidden_states.clone()
                new_hidden[:, 0, :] = (
                    frozen_latents[layer_idx, latent_step, :].unsqueeze(0).to(device)
                )
                return (new_hidden,) + output[1:]
            else:
                new_hidden = output.clone()
                new_hidden[:, 0, :] = (
                    frozen_latents[layer_idx, latent_step, :].unsqueeze(0).to(device)
                )
                return new_hidden

        return hook

    batch_size = input_ids.size(0)
    past_key_values = None

    with torch.no_grad():
        # Prefill with mean ablation
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )
        past_key_values = outputs.past_key_values

        for h in patch_handles:
            h.remove()

        # Latent loop with frozen activations
        latent_embd = frozen_latents_dict["initial_latent_embd"].to(device)

        for i in range(num_latents):
            for layer_idx, layer in enumerate(layers):
                hook = get_frozen_latent_hook(i, layer_idx)
                frozen_latent_handles.append(layer.register_forward_hook(hook))

            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )

            for h in frozen_latent_handles:
                h.remove()
            frozen_latent_handles = []

            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        # Generate from eocot token
        eot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")
        eot_ids = torch.tensor([[eot_id]], dtype=torch.long, device=device)
        eot_emb = model.get_embd(model.codi, model.model_name)(eot_ids).to(device)

        outputs = model.codi(
            inputs_embeds=eot_emb,
            output_hidden_states=False,
            attention_mask=None,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values
        logits_at_eocot = outputs.logits[:, -1, : model.codi.config.vocab_size - 1]

        # Token generation
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        sequences = input_ids.clone()
        output = eot_emb

        for step in range(max_new_tokens):
            out = model.codi(
                inputs_embeds=output,
                output_hidden_states=False,
                attention_mask=None,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, : model.codi.config.vocab_size - 1]

            if greedy:
                next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1)

            sequences = torch.cat(
                [sequences, next_token_ids.expand(batch_size, -1)], dim=1
            )

            for batch_idx in range(batch_size):
                if not finished[batch_idx]:
                    token_id = next_token_ids[batch_idx, 0].item()
                    if token_id == tokenizer.eos_token_id:
                        finished[batch_idx] = True

            if finished.all():
                break

            next_token_emb = model.get_embd(model.codi, model.model_name)(
                next_token_ids
            ).to(device)
            output = next_token_emb

        return {"sequences": sequences, "logits_at_eocot": logits_at_eocot}


def generate_with_mean_ablated_prompt_regenerated_latents(
    model,
    tokenizer,
    prompt,
    mean_activations_dict,
    max_new_tokens=128,
    num_latent_iterations=6,
    temperature=0.1,
    greedy=True,
):
    """
    Condition 3: Mean-ablated prompt + regenerated latents.
    Mean-ablates prompt activations, then lets latents regenerate normally.
    """
    device = model.codi.device
    layers = get_transformer_layers(model)

    mean_prefill = mean_activations_dict["mean_prefill"].to(device)
    max_seq_len = mean_prefill.shape[1]
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)
    prompt_seq_len = input_ids.shape[1] - 1

    # Set up mean ablation hooks
    patch_handles = []
    for layer_idx, layer in enumerate(layers):
        hook = _create_mean_ablation_hook(
            layer_idx, mean_prefill, prompt_seq_len, max_seq_len, device
        )
        patch_handles.append(layer.register_forward_hook(hook))

    batch_size = input_ids.size(0)
    past_key_values = None

    with torch.no_grad():
        # Prefill with mean ablation
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )
        past_key_values = outputs.past_key_values

        for h in patch_handles:
            h.remove()

        # Latent loop - regenerate normally
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        if model.use_prj:
            latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        for i in range(num_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )

            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        # Generate from eocot token
        eot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")
        eot_ids = torch.tensor([[eot_id]], dtype=torch.long, device=device)
        eot_emb = model.get_embd(model.codi, model.model_name)(eot_ids).to(device)
        output = eot_emb

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        sequences = input_ids.clone()

        for step in range(max_new_tokens):
            out = model.codi(
                inputs_embeds=output,
                output_hidden_states=False,
                attention_mask=None,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, : model.codi.config.vocab_size - 1]

            if greedy:
                next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1)

            sequences = torch.cat(
                [sequences, next_token_ids.expand(batch_size, -1)], dim=1
            )

            for batch_idx in range(batch_size):
                if not finished[batch_idx]:
                    token_id = next_token_ids[batch_idx, 0].item()
                    if token_id == tokenizer.eos_token_id:
                        finished[batch_idx] = True

            if finished.all():
                break

            next_token_emb = model.get_embd(model.codi, model.model_name)(
                next_token_ids
            ).to(device)
            output = next_token_emb

        return {"sequences": sequences}


def generate_with_mean_ablated_prompt_patched_latent_embeddings(
    model,
    tokenizer,
    prompt,
    mean_activations_dict,
    latent_embeddings_dict,
    max_new_tokens=128,
    num_latent_iterations=6,
    temperature=0.1,
    greedy=True,
):
    """
    Condition 4: Mean-ablated prompt + patched latent embeddings.
    Mean-ablates prompt activations, patches in original latent embeddings,
    but lets layer activations compute normally.
    """
    device = model.codi.device
    layers = get_transformer_layers(model)

    mean_prefill = mean_activations_dict["mean_prefill"].to(device)
    max_seq_len = mean_prefill.shape[1]
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)
    prompt_seq_len = input_ids.shape[1] - 1

    latent_embeddings = latent_embeddings_dict["latent_embeddings"]

    # Set up mean ablation hooks
    patch_handles = []
    for layer_idx, layer in enumerate(layers):
        hook = _create_mean_ablation_hook(
            layer_idx, mean_prefill, prompt_seq_len, max_seq_len, device
        )
        patch_handles.append(layer.register_forward_hook(hook))

    batch_size = input_ids.size(0)
    past_key_values = None

    with torch.no_grad():
        # Prefill with mean ablation
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )
        past_key_values = outputs.past_key_values

        for h in patch_handles:
            h.remove()

        # Latent loop - patch embeddings but compute activations normally
        latent_embd = latent_embeddings[0].to(device)

        for i in range(num_latent_iterations):
            if i < len(latent_embeddings):
                latent_embd_input = latent_embeddings[i].to(device)
            else:
                latent_embd_input = latent_embd

            outputs = model.codi(
                inputs_embeds=latent_embd_input,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )

            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        # Generate from eocot token
        eot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")
        eot_ids = torch.tensor([[eot_id]], dtype=torch.long, device=device)
        eot_emb = model.get_embd(model.codi, model.model_name)(eot_ids).to(device)
        output = eot_emb

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        sequences = input_ids.clone()

        for step in range(max_new_tokens):
            out = model.codi(
                inputs_embeds=output,
                output_hidden_states=False,
                attention_mask=None,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, : model.codi.config.vocab_size - 1]

            if greedy:
                next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1)

            sequences = torch.cat(
                [sequences, next_token_ids.expand(batch_size, -1)], dim=1
            )

            for batch_idx in range(batch_size):
                if not finished[batch_idx]:
                    token_id = next_token_ids[batch_idx, 0].item()
                    if token_id == tokenizer.eos_token_id:
                        finished[batch_idx] = True

            if finished.all():
                break

            next_token_emb = model.get_embd(model.codi, model.model_name)(
                next_token_ids
            ).to(device)
            output = next_token_emb

        return {"sequences": sequences}


def generate_with_mean_ablated_prompt_cross_template_mean_embeddings(
    model,
    tokenizer,
    prompt,
    mean_activations_dict,
    cross_template_mean_embeddings_dict,
    max_new_tokens=128,
    num_latent_iterations=6,
    temperature=0.1,
    greedy=True,
):
    """
    Condition: Mean-ablated prompt + cross-template mean latent embeddings.
    Mean-ablates prompt activations, patches in mean latent embeddings computed
    across other templates (with same X, Y, Z), but lets layer activations compute normally.
    """
    device = model.codi.device
    layers = get_transformer_layers(model)

    mean_prefill = mean_activations_dict["mean_prefill"].to(device)
    max_seq_len = mean_prefill.shape[1]
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)
    prompt_seq_len = input_ids.shape[1] - 1

    latent_embeddings = cross_template_mean_embeddings_dict["latent_embeddings"]

    # Set up mean ablation hooks
    patch_handles = []
    for layer_idx, layer in enumerate(layers):
        hook = _create_mean_ablation_hook(
            layer_idx, mean_prefill, prompt_seq_len, max_seq_len, device
        )
        patch_handles.append(layer.register_forward_hook(hook))

    batch_size = input_ids.size(0)
    past_key_values = None

    with torch.no_grad():
        # Prefill with mean ablation
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )
        past_key_values = outputs.past_key_values

        for h in patch_handles:
            h.remove()

        # Latent loop - patch embeddings but compute activations normally
        latent_embd = latent_embeddings[0].to(device)

        for i in range(num_latent_iterations):
            if i < len(latent_embeddings):
                latent_embd_input = latent_embeddings[i].to(device)
            else:
                latent_embd_input = latent_embd

            outputs = model.codi(
                inputs_embeds=latent_embd_input,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )

            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        # Generate from eocot token
        eot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")
        eot_ids = torch.tensor([[eot_id]], dtype=torch.long, device=device)
        eot_emb = model.get_embd(model.codi, model.model_name)(eot_ids).to(device)
        output = eot_emb

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        sequences = input_ids.clone()

        for step in range(max_new_tokens):
            out = model.codi(
                inputs_embeds=output,
                output_hidden_states=False,
                attention_mask=None,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, : model.codi.config.vocab_size - 1]

            if greedy:
                next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1)

            sequences = torch.cat(
                [sequences, next_token_ids.expand(batch_size, -1)], dim=1
            )

            for batch_idx in range(batch_size):
                if not finished[batch_idx]:
                    token_id = next_token_ids[batch_idx, 0].item()
                    if token_id == tokenizer.eos_token_id:
                        finished[batch_idx] = True

            if finished.all():
                break

            next_token_emb = model.get_embd(model.codi, model.model_name)(
                next_token_ids
            ).to(device)
            output = next_token_emb

        return {"sequences": sequences}


def generate_with_template_prompt_patched_embeddings(
    model,
    tokenizer,
    template_prompt,
    latent_embeddings_dict,
    max_new_tokens=128,
    num_latent_iterations=6,
    temperature=0.1,
    greedy=True,
):
    """
    Template baseline with patched embeddings.
    Uses template prompt with X, Y, Z placeholders, patches in latent embeddings from concrete prompt.
    """
    device = model.codi.device
    input_ids, attention_mask = prepare_inputs(model, tokenizer, template_prompt)
    latent_embeddings = latent_embeddings_dict["latent_embeddings"]

    batch_size = input_ids.size(0)
    past_key_values = None

    with torch.no_grad():
        # Prefill with template prompt
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )
        past_key_values = outputs.past_key_values

        # Latent loop - patch embeddings from concrete prompt
        latent_embd = latent_embeddings[0].to(device)

        for i in range(num_latent_iterations):
            if i < len(latent_embeddings):
                latent_embd_input = latent_embeddings[i].to(device)
            else:
                latent_embd_input = latent_embd

            outputs = model.codi(
                inputs_embeds=latent_embd_input,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )

            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        # Generate from eocot token
        eot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")
        eot_ids = torch.tensor([[eot_id]], dtype=torch.long, device=device)
        eot_emb = model.get_embd(model.codi, model.model_name)(eot_ids).to(device)
        output = eot_emb

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        sequences = input_ids.clone()

        for step in range(max_new_tokens):
            out = model.codi(
                inputs_embeds=output,
                output_hidden_states=False,
                attention_mask=None,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, : model.codi.config.vocab_size - 1]

            if greedy:
                next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1)

            sequences = torch.cat(
                [sequences, next_token_ids.expand(batch_size, -1)], dim=1
            )

            for batch_idx in range(batch_size):
                if not finished[batch_idx]:
                    token_id = next_token_ids[batch_idx, 0].item()
                    if token_id == tokenizer.eos_token_id:
                        finished[batch_idx] = True

            if finished.all():
                break

            next_token_emb = model.get_embd(model.codi, model.model_name)(
                next_token_ids
            ).to(device)
            output = next_token_emb

        return {"sequences": sequences}


def generate_with_template_prompt_cross_template_mean_embeddings(
    model,
    tokenizer,
    template_prompt,
    cross_template_mean_embeddings_dict,
    max_new_tokens=128,
    num_latent_iterations=6,
    temperature=0.1,
    greedy=True,
):
    """
    Template baseline with cross-template mean patched embeddings.
    Uses template prompt with X, Y, Z placeholders, patches in cross-template mean
    latent embeddings (averaged across other templates with same X, Y, Z).
    """
    device = model.codi.device
    input_ids, attention_mask = prepare_inputs(model, tokenizer, template_prompt)
    latent_embeddings = cross_template_mean_embeddings_dict["latent_embeddings"]

    batch_size = input_ids.size(0)
    past_key_values = None

    with torch.no_grad():
        # Prefill with template prompt
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )
        past_key_values = outputs.past_key_values

        # Latent loop - patch embeddings from cross-template mean
        latent_embd = latent_embeddings[0].to(device)

        for i in range(num_latent_iterations):
            if i < len(latent_embeddings):
                latent_embd_input = latent_embeddings[i].to(device)
            else:
                latent_embd_input = latent_embd

            outputs = model.codi(
                inputs_embeds=latent_embd_input,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )

            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        # Generate from eocot token
        eot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")
        eot_ids = torch.tensor([[eot_id]], dtype=torch.long, device=device)
        eot_emb = model.get_embd(model.codi, model.model_name)(eot_ids).to(device)
        output = eot_emb

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        sequences = input_ids.clone()

        for step in range(max_new_tokens):
            out = model.codi(
                inputs_embeds=output,
                output_hidden_states=False,
                attention_mask=None,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, : model.codi.config.vocab_size - 1]

            if greedy:
                next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1)

            sequences = torch.cat(
                [sequences, next_token_ids.expand(batch_size, -1)], dim=1
            )

            for batch_idx in range(batch_size):
                if not finished[batch_idx]:
                    token_id = next_token_ids[batch_idx, 0].item()
                    if token_id == tokenizer.eos_token_id:
                        finished[batch_idx] = True

            if finished.all():
                break

            next_token_emb = model.get_embd(model.codi, model.model_name)(
                next_token_ids
            ).to(device)
            output = next_token_emb

        return {"sequences": sequences}


def generate_with_template_prompt_frozen_latents(
    model,
    tokenizer,
    template_prompt,
    frozen_latents_dict,
    max_new_tokens=128,
    temperature=0.1,
    greedy=True,
):
    """
    Template baseline with frozen latents.
    Uses template prompt with X, Y, Z placeholders, freezes latent activations from concrete prompt.
    """
    device = model.codi.device
    layers = get_transformer_layers(model)
    num_layers = len(layers)

    input_ids, attention_mask = prepare_inputs(model, tokenizer, template_prompt)
    frozen_latents = frozen_latents_dict["latent_activations"].to(device)
    num_latents = frozen_latents.shape[1]
    frozen_latent_handles = []

    def get_frozen_latent_hook(latent_step, layer_idx):
        def hook(module, args, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                new_hidden = hidden_states.clone()
                new_hidden[:, 0, :] = (
                    frozen_latents[layer_idx, latent_step, :].unsqueeze(0).to(device)
                )
                return (new_hidden,) + output[1:]
            else:
                new_hidden = output.clone()
                new_hidden[:, 0, :] = (
                    frozen_latents[layer_idx, latent_step, :].unsqueeze(0).to(device)
                )
                return new_hidden

        return hook

    batch_size = input_ids.size(0)
    past_key_values = None

    with torch.no_grad():
        # Prefill with template prompt
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )
        past_key_values = outputs.past_key_values

        # Latent loop with frozen activations from concrete prompt
        latent_embd = frozen_latents_dict["initial_latent_embd"].to(device)

        for i in range(num_latents):
            for layer_idx, layer in enumerate(layers):
                hook = get_frozen_latent_hook(i, layer_idx)
                frozen_latent_handles.append(layer.register_forward_hook(hook))

            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )

            for h in frozen_latent_handles:
                h.remove()
            frozen_latent_handles = []

            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        # Generate from eocot token
        eot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")
        eot_ids = torch.tensor([[eot_id]], dtype=torch.long, device=device)
        eot_emb = model.get_embd(model.codi, model.model_name)(eot_ids).to(device)

        outputs = model.codi(
            inputs_embeds=eot_emb,
            output_hidden_states=False,
            attention_mask=None,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values
        logits_at_eocot = outputs.logits[:, -1, : model.codi.config.vocab_size - 1]

        # Token generation
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        sequences = input_ids.clone()
        output = eot_emb

        for step in range(max_new_tokens):
            out = model.codi(
                inputs_embeds=output,
                output_hidden_states=False,
                attention_mask=None,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, : model.codi.config.vocab_size - 1]

            if greedy:
                next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1)

            sequences = torch.cat(
                [sequences, next_token_ids.expand(batch_size, -1)], dim=1
            )

            for batch_idx in range(batch_size):
                if not finished[batch_idx]:
                    token_id = next_token_ids[batch_idx, 0].item()
                    if token_id == tokenizer.eos_token_id:
                        finished[batch_idx] = True

            if finished.all():
                break

            next_token_emb = model.get_embd(model.codi, model.model_name)(
                next_token_ids
            ).to(device)
            output = next_token_emb

        return {"sequences": sequences, "logits_at_eocot": logits_at_eocot}


# ============================================================================
# Main Experiment
# ============================================================================


def generate_test_cases(template, get_answer, combinations):
    """Generate test cases from specific combinations."""
    test_cases = []
    for i, (X, Y, Z) in enumerate(combinations):
        prompt = template.format(X=X, Y=Y, Z=Z)
        ground_truth = get_answer(X, Y, Z)
        test_cases.append(
            {
                "id": i,
                "X": X,
                "Y": Y,
                "Z": Z,
                "prompt": prompt,
                "ground_truth": ground_truth,
            }
        )
    return test_cases


def _get_templates():
    """Get the list of all available templates."""
    return [
        "A team starts with {X} members. They recruit {Y} new members. Then each current member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else.",
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


def get_num_templates():
    """Get the number of available templates."""
    return len(_get_templates())


def get_template_abstract(template_idx):
    """Get abstract template string with placeholders instead of numbers."""
    templates = _get_templates()
    template_str = templates[template_idx]
    # Replace {X}, {Y}, {Z} with placeholders and remove all curly braces
    abstract_template = (
        template_str.replace("{X}", "X")
        .replace("{Y}", "Y")
        .replace("{Z}", "Z")
        .replace("{", "")
        .replace("}", "")
    )
    return abstract_template


def get_template_and_param_mapping(template_idx):
    """Get template string and parameter mapping function for a given template index."""
    templates = _get_templates()

    def get_params(x, y, z, idx):
        """Get mapped parameters for template index."""
        # All templates use the same calculation: (X+Y) * (1+Z)
        param_mappings = [
            (x, y, z),
            (x, y, z),
            (x, y, z),
            (x, y, z),
            (x, y, z),
            (x, y, z),
            (x, y, z),
            (x, y, z),
            (x, y, z),
            (x, y, z),
        ]
        return param_mappings[idx]

    return templates[template_idx], lambda x, y, z: get_params(x, y, z, template_idx)


def main():
    """Run the main experiment with multiple prompt templates."""
    load_dotenv()

    # Model setup
    print("Loading model...")
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
    tokenizer = model.tokenizer

    def get_answer(X, Y, Z):
        """Compute ground truth answer: (X+Y) * (1+Z)"""
        step_1 = X + Y
        step_2 = step_1 * Z
        step_3 = step_1 + step_2
        return step_3

    # Experiment parameters
    num_samples_per_prompt = 3
    temperature = 1.0
    greedy = False
    num_test_cases = 50
    num_mean_activation_samples = 50

    # Experiment configuration - enable/disable specific conditions
    ENABLED_CONDITIONS = {
        "baseline": True,
        "mean_ablated_frozen": False,
        "mean_ablated_regenerated": True,
        "mean_ablated_patched": True,
        "mean_ablated_cross_template_mean": True,
        "template_plain": True,
        "template_patched": True,
        "template_frozen": False,
        "template_cross_template_mean": True,
    }

    # Determine number of templates dynamically by trying to access them
    num_templates = get_num_templates()
    print(f"Found {num_templates} templates to process")
    print(f"Enabled conditions: {[k for k, v in ENABLED_CONDITIONS.items() if v]}")

    # Generate all possible combinations of base X, Y, Z in [1, 10]
    print("\nGenerating all possible combinations...")
    all_combinations = []
    for X in range(1, 11):
        for Y in range(1, 11):
            for Z in range(1, 11):
                all_combinations.append((X, Y, Z))

    print(f"Total combinations: {len(all_combinations)}")

    # Shuffle and split into disjoint sets
    np.random.seed(42)
    shuffled_combinations = np.array(all_combinations)
    np.random.shuffle(shuffled_combinations)

    # Split: first N for mean activations, next M for test cases
    mean_activation_combinations = shuffled_combinations[
        :num_mean_activation_samples
    ].tolist()
    test_case_combinations = shuffled_combinations[
        num_mean_activation_samples : num_mean_activation_samples + num_test_cases
    ].tolist()

    print(f"Mean activation samples: {len(mean_activation_combinations)}")
    print(f"Test case samples: {len(test_case_combinations)}")

    # Storage for results across all templates
    all_template_results = []
    all_template_baseline_accuracies = []

    # Process each template
    for template_idx in range(num_templates):
        print(f"\n{'=' * 80}")
        print(f"Processing Template {template_idx + 1}/{num_templates}")
        print(f"{'=' * 80}")

        # Get template string and parameter mapping
        template_str, param_mapping = get_template_and_param_mapping(template_idx)
        print(f"Template: {template_str[:80]}...")

        # Generate prompts for mean activation computation
        print("\nGenerating prompts for mean activation computation...")
        mean_activation_prompts = []
        for X, Y, Z in mean_activation_combinations:
            mapped_X, mapped_Y, mapped_Z = param_mapping(X, Y, Z)
            prompt = template_str.format(X=mapped_X, Y=mapped_Y, Z=mapped_Z)
            mean_activation_prompts.append(prompt)

        # Compute mean activations for this template
        print(f"Computing mean prompt activations for template {template_idx + 1}...")
        mean_activations_dict = compute_mean_prompt_activations(
            model,
            tokenizer,
            template_str,
            [(param_mapping(X, Y, Z)) for X, Y, Z in mean_activation_combinations],
        )

        # Generate test cases for this template
        print(f"\nGenerating test cases for template {template_idx + 1}...")
        template_test_cases = []
        for i, (X, Y, Z) in enumerate(test_case_combinations):
            mapped_X, mapped_Y, mapped_Z = param_mapping(X, Y, Z)
            prompt = template_str.format(X=mapped_X, Y=mapped_Y, Z=mapped_Z)
            ground_truth = get_answer(X, Y, Z)
            print(f"Prompt: {prompt}")
            print(f"Ground truth: {ground_truth}")
            template_test_cases.append(
                {
                    "id": i,
                    "base_X": X,
                    "base_Y": Y,
                    "base_Z": Z,
                    "mapped_X": mapped_X,
                    "mapped_Y": mapped_Y,
                    "mapped_Z": mapped_Z,
                    "prompt": prompt,
                    "ground_truth": ground_truth,
                }
            )

        # Get abstract template for template-based conditions
        template_abstract = get_template_abstract(template_idx)

        # Run baseline experiment for this template
        print(f"\nRunning experiment for template {template_idx + 1}...")
        template_results = []
        baseline_position_correct = [[] for _ in range(num_samples_per_prompt)]
        frozen_position_correct = [[] for _ in range(num_samples_per_prompt)]
        regenerated_position_correct = [[] for _ in range(num_samples_per_prompt)]
        patched_position_correct = [[] for _ in range(num_samples_per_prompt)]
        cross_template_mean_position_correct = [
            [] for _ in range(num_samples_per_prompt)
        ]
        template_plain_position_correct = [[] for _ in range(num_samples_per_prompt)]
        template_patched_position_correct = [[] for _ in range(num_samples_per_prompt)]
        template_frozen_position_correct = [[] for _ in range(num_samples_per_prompt)]
        template_cross_template_mean_position_correct = [
            [] for _ in range(num_samples_per_prompt)
        ]

        for test_case in tqdm(template_test_cases, desc=f"Template {template_idx + 1}"):
            prompt = test_case["prompt"]
            ground_truth = test_case["ground_truth"]

            try:
                # Capture data for interventions (only if needed)
                frozen_latents_dict = None
                latent_embeddings_dict = None

                if (
                    ENABLED_CONDITIONS["mean_ablated_frozen"]
                    or ENABLED_CONDITIONS["template_frozen"]
                ):
                    frozen_latents_dict = capture_target_frozen_latents(
                        model, tokenizer, prompt, num_latents=6
                    )

                if (
                    ENABLED_CONDITIONS["mean_ablated_patched"]
                    or ENABLED_CONDITIONS["template_patched"]
                ):
                    latent_embeddings_dict = capture_latent_embeddings(
                        model, tokenizer, prompt, num_latents=6
                    )

                # Compute cross-template mean embeddings if needed
                cross_template_mean_embeddings_dict = None
                if (
                    ENABLED_CONDITIONS["mean_ablated_cross_template_mean"]
                    or ENABLED_CONDITIONS["template_cross_template_mean"]
                ):
                    cross_template_mean_embeddings_dict = (
                        compute_mean_latent_embeddings_across_other_templates(
                            model,
                            tokenizer,
                            template_idx,
                            test_case["base_X"],
                            test_case["base_Y"],
                            test_case["base_Z"],
                            num_latents=6,
                        )
                    )

                # Storage for this prompt's samples
                prompt_results = {
                    "id": test_case["id"],
                    "base_X": test_case["base_X"],
                    "base_Y": test_case["base_Y"],
                    "base_Z": test_case["base_Z"],
                    "mapped_X": test_case["mapped_X"],
                    "mapped_Y": test_case["mapped_Y"],
                    "mapped_Z": test_case["mapped_Z"],
                    "ground_truth": ground_truth,
                    "baseline_samples": [],
                    "frozen_samples": [],
                    "regenerated_samples": [],
                    "patched_samples": [],
                    "cross_template_mean_samples": [],
                    "template_plain_samples": [],
                    "template_patched_samples": [],
                    "template_frozen_samples": [],
                    "template_cross_template_mean_samples": [],
                }

                # Generate multiple samples for each condition
                for sample_idx in range(num_samples_per_prompt):
                    # Condition 1: Baseline
                    if ENABLED_CONDITIONS["baseline"]:
                        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                            model.codi.device
                        )
                        attention_mask = tokenizer(
                            prompt, return_tensors="pt"
                        ).attention_mask.to(model.codi.device)
                        output_baseline = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=128,
                            num_latent_iterations=6,
                            temperature=temperature,
                            greedy=greedy,
                            return_latent_vectors=False,
                            remove_eos=False,
                            output_attentions=False,
                            skip_thinking=False,
                            output_hidden_states=True,
                            sot_token=tokenizer.convert_tokens_to_ids("<|bocot|>"),
                            eot_token=tokenizer.convert_tokens_to_ids("<|eocot|>"),
                            verbalize_cot=False,
                        )
                        generated_text_baseline = tokenizer.decode(
                            output_baseline["sequences"][0], skip_special_tokens=False
                        )
                        baseline_answer = extract_answer_number(generated_text_baseline)
                        baseline_correct = (
                            baseline_answer is not None
                            and baseline_answer != float("inf")
                            and int(baseline_answer) == ground_truth
                        )
                        prompt_results["baseline_samples"].append(
                            {
                                "answer": baseline_answer,
                                "correct": baseline_correct,
                                "text": generated_text_baseline,
                            }
                        )
                        baseline_position_correct[sample_idx].append(baseline_correct)

                    # Condition 2: Mean-ablated + frozen latents
                    if ENABLED_CONDITIONS["mean_ablated_frozen"]:
                        output_mean_ablated_frozen = (
                            generate_with_mean_ablated_prompt_frozen_latents(
                                model,
                                tokenizer,
                                prompt,
                                mean_activations_dict=mean_activations_dict,
                                frozen_latents_dict=frozen_latents_dict,
                                max_new_tokens=128,
                                temperature=temperature,
                                greedy=greedy,
                            )
                        )
                        generated_text_mean_ablated_frozen = tokenizer.decode(
                            output_mean_ablated_frozen["sequences"][0],
                            skip_special_tokens=False,
                        )
                        mean_ablated_frozen_answer = extract_answer_number(
                            generated_text_mean_ablated_frozen
                        )
                        frozen_correct = (
                            mean_ablated_frozen_answer is not None
                            and mean_ablated_frozen_answer != float("inf")
                            and int(mean_ablated_frozen_answer) == ground_truth
                        )
                        prompt_results["frozen_samples"].append(
                            {
                                "answer": mean_ablated_frozen_answer,
                                "correct": frozen_correct,
                                "text": generated_text_mean_ablated_frozen,
                            }
                        )
                        frozen_position_correct[sample_idx].append(frozen_correct)

                    # Condition 3: Mean-ablated + regenerated latents
                    if ENABLED_CONDITIONS["mean_ablated_regenerated"]:
                        output_mean_ablated_regenerated = (
                            generate_with_mean_ablated_prompt_regenerated_latents(
                                model,
                                tokenizer,
                                prompt,
                                mean_activations_dict=mean_activations_dict,
                                max_new_tokens=128,
                                num_latent_iterations=6,
                                temperature=temperature,
                                greedy=greedy,
                            )
                        )
                        generated_text_mean_ablated_regenerated = tokenizer.decode(
                            output_mean_ablated_regenerated["sequences"][0],
                            skip_special_tokens=False,
                        )
                        regenerated_answer = extract_answer_number(
                            generated_text_mean_ablated_regenerated
                        )
                        regenerated_correct = (
                            regenerated_answer is not None
                            and regenerated_answer != float("inf")
                            and int(regenerated_answer) == ground_truth
                        )
                        prompt_results["regenerated_samples"].append(
                            {
                                "answer": regenerated_answer,
                                "correct": regenerated_correct,
                                "text": generated_text_mean_ablated_regenerated,
                            }
                        )
                        regenerated_position_correct[sample_idx].append(
                            regenerated_correct
                        )

                    # Condition 4: Mean-ablated + patched embeddings
                    if ENABLED_CONDITIONS["mean_ablated_patched"]:
                        output_mean_ablated_patched_embeddings = (
                            generate_with_mean_ablated_prompt_patched_latent_embeddings(
                                model,
                                tokenizer,
                                prompt,
                                mean_activations_dict=mean_activations_dict,
                                latent_embeddings_dict=latent_embeddings_dict,
                                max_new_tokens=128,
                                num_latent_iterations=6,
                                temperature=temperature,
                                greedy=greedy,
                            )
                        )
                        generated_text_mean_ablated_patched_embeddings = (
                            tokenizer.decode(
                                output_mean_ablated_patched_embeddings["sequences"][0],
                                skip_special_tokens=False,
                            )
                        )
                        patched_answer = extract_answer_number(
                            generated_text_mean_ablated_patched_embeddings
                        )
                        patched_correct = (
                            patched_answer is not None
                            and patched_answer != float("inf")
                            and int(patched_answer) == ground_truth
                        )
                        prompt_results["patched_samples"].append(
                            {
                                "answer": patched_answer,
                                "correct": patched_correct,
                                "text": generated_text_mean_ablated_patched_embeddings,
                            }
                        )
                        patched_position_correct[sample_idx].append(patched_correct)

                    # Condition: Mean-ablated + cross-template mean embeddings
                    if ENABLED_CONDITIONS["mean_ablated_cross_template_mean"]:
                        output_cross_template_mean = generate_with_mean_ablated_prompt_cross_template_mean_embeddings(
                            model,
                            tokenizer,
                            prompt,
                            mean_activations_dict=mean_activations_dict,
                            cross_template_mean_embeddings_dict=cross_template_mean_embeddings_dict,
                            max_new_tokens=128,
                            num_latent_iterations=6,
                            temperature=temperature,
                            greedy=greedy,
                        )
                        generated_text_cross_template_mean = tokenizer.decode(
                            output_cross_template_mean["sequences"][0],
                            skip_special_tokens=False,
                        )
                        cross_template_mean_answer = extract_answer_number(
                            generated_text_cross_template_mean
                        )
                        cross_template_mean_correct = (
                            cross_template_mean_answer is not None
                            and cross_template_mean_answer != float("inf")
                            and int(cross_template_mean_answer) == ground_truth
                        )
                        prompt_results["cross_template_mean_samples"].append(
                            {
                                "answer": cross_template_mean_answer,
                                "correct": cross_template_mean_correct,
                                "text": generated_text_cross_template_mean,
                            }
                        )
                        cross_template_mean_position_correct[sample_idx].append(
                            cross_template_mean_correct
                        )

                    # Condition 5: Template prompt (plain) - with placeholders
                    if ENABLED_CONDITIONS["template_plain"]:
                        input_ids_template = tokenizer(
                            template_abstract, return_tensors="pt"
                        ).input_ids.to(model.codi.device)
                        attention_mask_template = tokenizer(
                            template_abstract, return_tensors="pt"
                        ).attention_mask.to(model.codi.device)
                        output_template_plain = model.generate(
                            input_ids=input_ids_template,
                            attention_mask=attention_mask_template,
                            max_new_tokens=128,
                            num_latent_iterations=6,
                            temperature=temperature,
                            greedy=greedy,
                            return_latent_vectors=False,
                            remove_eos=False,
                            output_attentions=False,
                            skip_thinking=False,
                            output_hidden_states=True,
                            sot_token=tokenizer.convert_tokens_to_ids("<|bocot|>"),
                            eot_token=tokenizer.convert_tokens_to_ids("<|eocot|>"),
                            verbalize_cot=False,
                        )
                        generated_text_template_plain = tokenizer.decode(
                            output_template_plain["sequences"][0],
                            skip_special_tokens=False,
                        )
                        template_plain_answer = extract_answer_number(
                            generated_text_template_plain
                        )
                        template_plain_correct = (
                            template_plain_answer is not None
                            and template_plain_answer != float("inf")
                            and int(template_plain_answer) == ground_truth
                        )
                        prompt_results["template_plain_samples"].append(
                            {
                                "answer": template_plain_answer,
                                "correct": template_plain_correct,
                                "text": generated_text_template_plain,
                            }
                        )
                        template_plain_position_correct[sample_idx].append(
                            template_plain_correct
                        )

                    # Condition 6: Template prompt + patched embeddings
                    if ENABLED_CONDITIONS["template_patched"]:
                        output_template_patched = (
                            generate_with_template_prompt_patched_embeddings(
                                model,
                                tokenizer,
                                template_abstract,
                                latent_embeddings_dict=latent_embeddings_dict,
                                max_new_tokens=128,
                                num_latent_iterations=6,
                                temperature=temperature,
                                greedy=greedy,
                            )
                        )
                        generated_text_template_patched = tokenizer.decode(
                            output_template_patched["sequences"][0],
                            skip_special_tokens=False,
                        )
                        template_patched_answer = extract_answer_number(
                            generated_text_template_patched
                        )
                        template_patched_correct = (
                            template_patched_answer is not None
                            and template_patched_answer != float("inf")
                            and int(template_patched_answer) == ground_truth
                        )
                        prompt_results["template_patched_samples"].append(
                            {
                                "answer": template_patched_answer,
                                "correct": template_patched_correct,
                                "text": generated_text_template_patched,
                            }
                        )
                        template_patched_position_correct[sample_idx].append(
                            template_patched_correct
                        )

                    # Condition 7: Template prompt + frozen latents
                    if ENABLED_CONDITIONS["template_frozen"]:
                        output_template_frozen = (
                            generate_with_template_prompt_frozen_latents(
                                model,
                                tokenizer,
                                template_abstract,
                                frozen_latents_dict=frozen_latents_dict,
                                max_new_tokens=128,
                                temperature=temperature,
                                greedy=greedy,
                            )
                        )
                        generated_text_template_frozen = tokenizer.decode(
                            output_template_frozen["sequences"][0],
                            skip_special_tokens=False,
                        )
                        template_frozen_answer = extract_answer_number(
                            generated_text_template_frozen
                        )
                        template_frozen_correct = (
                            template_frozen_answer is not None
                            and template_frozen_answer != float("inf")
                            and int(template_frozen_answer) == ground_truth
                        )
                        prompt_results["template_frozen_samples"].append(
                            {
                                "answer": template_frozen_answer,
                                "correct": template_frozen_correct,
                                "text": generated_text_template_frozen,
                            }
                        )
                        template_frozen_position_correct[sample_idx].append(
                            template_frozen_correct
                        )

                    # Condition: Template prompt + cross-template mean embeddings
                    if ENABLED_CONDITIONS["template_cross_template_mean"]:
                        output_template_cross_template_mean = generate_with_template_prompt_cross_template_mean_embeddings(
                            model,
                            tokenizer,
                            template_abstract,
                            cross_template_mean_embeddings_dict=cross_template_mean_embeddings_dict,
                            max_new_tokens=128,
                            num_latent_iterations=6,
                            temperature=temperature,
                            greedy=greedy,
                        )
                        generated_text_template_cross_template_mean = tokenizer.decode(
                            output_template_cross_template_mean["sequences"][0],
                            skip_special_tokens=False,
                        )
                        template_cross_template_mean_answer = extract_answer_number(
                            generated_text_template_cross_template_mean
                        )
                        template_cross_template_mean_correct = (
                            template_cross_template_mean_answer is not None
                            and template_cross_template_mean_answer != float("inf")
                            and int(template_cross_template_mean_answer) == ground_truth
                        )
                        prompt_results["template_cross_template_mean_samples"].append(
                            {
                                "answer": template_cross_template_mean_answer,
                                "correct": template_cross_template_mean_correct,
                                "text": generated_text_template_cross_template_mean,
                            }
                        )
                        template_cross_template_mean_position_correct[
                            sample_idx
                        ].append(template_cross_template_mean_correct)

                template_results.append(prompt_results)

            except Exception as e:
                print(f"\nError processing test case {test_case['id']}: {e}")
                import traceback

                traceback.print_exc()
                template_results.append(
                    {
                        "id": test_case["id"],
                        "base_X": test_case["base_X"],
                        "base_Y": test_case["base_Y"],
                        "base_Z": test_case["base_Z"],
                        "ground_truth": ground_truth,
                        "error": str(e),
                    }
                )

        # Calculate accuracies for enabled conditions only
        def calc_accuracies(position_correct):
            """Calculate position-wise accuracies from position_correct list."""
            position_accuracies = []
            for position in range(num_samples_per_prompt):
                if len(position_correct[position]) > 0:
                    position_accuracies.append(np.mean(position_correct[position]))
            mean_acc = np.mean(position_accuracies) if position_accuracies else 0.0
            std_acc = np.std(position_accuracies) if position_accuracies else 0.0
            return position_accuracies, mean_acc, std_acc

        (
            baseline_position_accuracies,
            baseline_mean_acc,
            baseline_std_acc,
        ) = (
            calc_accuracies(baseline_position_correct)
            if ENABLED_CONDITIONS["baseline"]
            else ([], 0.0, 0.0)
        )
        (
            frozen_position_accuracies,
            frozen_mean_acc,
            frozen_std_acc,
        ) = (
            calc_accuracies(frozen_position_correct)
            if ENABLED_CONDITIONS["mean_ablated_frozen"]
            else ([], 0.0, 0.0)
        )
        (
            regenerated_position_accuracies,
            regenerated_mean_acc,
            regenerated_std_acc,
        ) = (
            calc_accuracies(regenerated_position_correct)
            if ENABLED_CONDITIONS["mean_ablated_regenerated"]
            else ([], 0.0, 0.0)
        )
        (
            patched_position_accuracies,
            patched_mean_acc,
            patched_std_acc,
        ) = (
            calc_accuracies(patched_position_correct)
            if ENABLED_CONDITIONS["mean_ablated_patched"]
            else ([], 0.0, 0.0)
        )
        (
            cross_template_mean_position_accuracies,
            cross_template_mean_mean_acc,
            cross_template_mean_std_acc,
        ) = (
            calc_accuracies(cross_template_mean_position_correct)
            if ENABLED_CONDITIONS["mean_ablated_cross_template_mean"]
            else ([], 0.0, 0.0)
        )
        (
            template_plain_position_accuracies,
            template_plain_mean_acc,
            template_plain_std_acc,
        ) = (
            calc_accuracies(template_plain_position_correct)
            if ENABLED_CONDITIONS["template_plain"]
            else ([], 0.0, 0.0)
        )
        (
            template_patched_position_accuracies,
            template_patched_mean_acc,
            template_patched_std_acc,
        ) = (
            calc_accuracies(template_patched_position_correct)
            if ENABLED_CONDITIONS["template_patched"]
            else ([], 0.0, 0.0)
        )
        (
            template_frozen_position_accuracies,
            template_frozen_mean_acc,
            template_frozen_std_acc,
        ) = (
            calc_accuracies(template_frozen_position_correct)
            if ENABLED_CONDITIONS["template_frozen"]
            else ([], 0.0, 0.0)
        )
        (
            template_cross_template_mean_position_accuracies,
            template_cross_template_mean_mean_acc,
            template_cross_template_mean_std_acc,
        ) = (
            calc_accuracies(template_cross_template_mean_position_correct)
            if ENABLED_CONDITIONS["template_cross_template_mean"]
            else ([], 0.0, 0.0)
        )

        print(f"\nTemplate {template_idx + 1} Results:")
        print("  Mean Accuracies:")
        print(
            f"    Baseline:          {baseline_mean_acc:.2%} ± {baseline_std_acc:.2%}"
        )
        print(f"    Frozen:            {frozen_mean_acc:.2%} ± {frozen_std_acc:.2%}")
        print(
            f"    Regenerated:       {regenerated_mean_acc:.2%} ± {regenerated_std_acc:.2%}"
        )
        print(f"    Patched:           {patched_mean_acc:.2%} ± {patched_std_acc:.2%}")
        print(
            f"    Cross-Template Mean: {cross_template_mean_mean_acc:.2%} ± {cross_template_mean_std_acc:.2%}"
        )
        print(
            f"    Template Plain:    {template_plain_mean_acc:.2%} ± {template_plain_std_acc:.2%}"
        )
        print(
            f"    Template Patched:  {template_patched_mean_acc:.2%} ± {template_patched_std_acc:.2%}"
        )
        print(
            f"    Template Frozen:   {template_frozen_mean_acc:.2%} ± {template_frozen_std_acc:.2%}"
        )
        print(
            f"    Template Cross-Tmpl Mean: {template_cross_template_mean_mean_acc:.2%} ± {template_cross_template_mean_std_acc:.2%}"
        )

        # Store results for this template
        all_template_results.append(
            {
                "template_idx": template_idx,
                "template_str": template_str,
                "baseline_mean_acc": baseline_mean_acc,
                "baseline_std_acc": baseline_std_acc,
                "baseline_position_accuracies": baseline_position_accuracies,
                "frozen_mean_acc": frozen_mean_acc,
                "frozen_std_acc": frozen_std_acc,
                "frozen_position_accuracies": frozen_position_accuracies,
                "regenerated_mean_acc": regenerated_mean_acc,
                "regenerated_std_acc": regenerated_std_acc,
                "regenerated_position_accuracies": regenerated_position_accuracies,
                "patched_mean_acc": patched_mean_acc,
                "patched_std_acc": patched_std_acc,
                "patched_position_accuracies": patched_position_accuracies,
                "cross_template_mean_mean_acc": cross_template_mean_mean_acc,
                "cross_template_mean_std_acc": cross_template_mean_std_acc,
                "cross_template_mean_position_accuracies": cross_template_mean_position_accuracies,
                "template_plain_mean_acc": template_plain_mean_acc,
                "template_plain_std_acc": template_plain_std_acc,
                "template_plain_position_accuracies": template_plain_position_accuracies,
                "template_patched_mean_acc": template_patched_mean_acc,
                "template_patched_std_acc": template_patched_std_acc,
                "template_patched_position_accuracies": template_patched_position_accuracies,
                "template_frozen_mean_acc": template_frozen_mean_acc,
                "template_frozen_std_acc": template_frozen_std_acc,
                "template_frozen_position_accuracies": template_frozen_position_accuracies,
                "template_cross_template_mean_mean_acc": template_cross_template_mean_mean_acc,
                "template_cross_template_mean_std_acc": template_cross_template_mean_std_acc,
                "template_cross_template_mean_position_accuracies": template_cross_template_mean_position_accuracies,
                "results": template_results,
            }
        )
        all_template_baseline_accuracies.append(baseline_mean_acc)

    # Calculate overall averages for all conditions
    overall_baseline_mean = np.mean(all_template_baseline_accuracies)
    overall_baseline_std = np.std(all_template_baseline_accuracies)

    all_frozen_accuracies = [tr["frozen_mean_acc"] for tr in all_template_results]
    all_regenerated_accuracies = [
        tr["regenerated_mean_acc"] for tr in all_template_results
    ]
    all_patched_accuracies = [tr["patched_mean_acc"] for tr in all_template_results]
    all_cross_template_mean_accuracies = [
        tr["cross_template_mean_mean_acc"] for tr in all_template_results
    ]
    all_template_plain_accuracies = [
        tr["template_plain_mean_acc"] for tr in all_template_results
    ]
    all_template_patched_accuracies = [
        tr["template_patched_mean_acc"] for tr in all_template_results
    ]
    all_template_frozen_accuracies = [
        tr["template_frozen_mean_acc"] for tr in all_template_results
    ]
    all_template_cross_template_mean_accuracies = [
        tr["template_cross_template_mean_mean_acc"] for tr in all_template_results
    ]

    overall_frozen_mean = np.mean(all_frozen_accuracies)
    overall_frozen_std = np.std(all_frozen_accuracies)
    overall_regenerated_mean = np.mean(all_regenerated_accuracies)
    overall_regenerated_std = np.std(all_regenerated_accuracies)
    overall_patched_mean = np.mean(all_patched_accuracies)
    overall_patched_std = np.std(all_patched_accuracies)
    overall_cross_template_mean_mean = np.mean(all_cross_template_mean_accuracies)
    overall_cross_template_mean_std = np.std(all_cross_template_mean_accuracies)
    overall_template_plain_mean = np.mean(all_template_plain_accuracies)
    overall_template_plain_std = np.std(all_template_plain_accuracies)
    overall_template_patched_mean = np.mean(all_template_patched_accuracies)
    overall_template_patched_std = np.std(all_template_patched_accuracies)
    overall_template_frozen_mean = np.mean(all_template_frozen_accuracies)
    overall_template_frozen_std = np.std(all_template_frozen_accuracies)
    overall_template_cross_template_mean_mean = np.mean(
        all_template_cross_template_mean_accuracies
    )
    overall_template_cross_template_mean_std = np.std(
        all_template_cross_template_mean_accuracies
    )

    print(f"\n{'=' * 80}")
    print("OVERALL RESULTS ACROSS ALL TEMPLATES:")
    print(f"{'=' * 80}")
    print("Mean Accuracies (averaged across all templates):")
    print(
        f"  Baseline:          {overall_baseline_mean:.2%} ± {overall_baseline_std:.2%}"
    )
    print(f"  Frozen:            {overall_frozen_mean:.2%} ± {overall_frozen_std:.2%}")
    print(
        f"  Regenerated:       {overall_regenerated_mean:.2%} ± {overall_regenerated_std:.2%}"
    )
    print(
        f"  Patched:           {overall_patched_mean:.2%} ± {overall_patched_std:.2%}"
    )
    print(
        f"  Cross-Template Mean: {overall_cross_template_mean_mean:.2%} ± {overall_cross_template_mean_std:.2%}"
    )
    print(
        f"  Template Plain:    {overall_template_plain_mean:.2%} ± {overall_template_plain_std:.2%}"
    )
    print(
        f"  Template Patched:  {overall_template_patched_mean:.2%} ± {overall_template_patched_std:.2%}"
    )
    print(
        f"  Template Frozen:   {overall_template_frozen_mean:.2%} ± {overall_template_frozen_std:.2%}"
    )
    print(
        f"  Template Cross-Tmpl Mean: {overall_template_cross_template_mean_mean:.2%} ± {overall_template_cross_template_mean_std:.2%}"
    )

    # Calculate position-wise accuracies across all templates
    if num_samples_per_prompt > 1:
        print("\nPosition-wise Accuracies (averaged across all templates):")
        all_position_accuracies = [[] for _ in range(num_samples_per_prompt)]
        for template_result in all_template_results:
            pos_accs = template_result["baseline_position_accuracies"]
            for pos_idx in range(len(pos_accs)):
                if pos_idx < len(all_position_accuracies):
                    all_position_accuracies[pos_idx].append(pos_accs[pos_idx])

        for pos in range(num_samples_per_prompt):
            if len(all_position_accuracies[pos]) > 0:
                pos_mean = np.mean(all_position_accuracies[pos])
                pos_std = np.std(all_position_accuracies[pos])
                print(f"  Position {pos + 1}: {pos_mean:.2%} ± {pos_std:.2%}")

    print("\nPer-Template Baseline Accuracies:")
    for i, (template_result, acc) in enumerate(
        zip(all_template_results, all_template_baseline_accuracies)
    ):
        print(f"  Template {i + 1}: {acc:.2%}")

    # Create visualizations
    print("\nCreating visualizations...")

    # Plot 1: All conditions averaged over all templates
    fig1, ax1 = plt.subplots(1, 1, figsize=(16, 6))
    methods = [
        "Baseline",
        "Mean-Abl\n+ Frozen",
        "Mean-Abl\n+ Regen",
        "Mean-Abl\n+ Patched",
        "Mean-Abl\n+ Cross-Tmpl",
        "Template\nPlain",
        "Template\n+ Patched",
        "Template\n+ Frozen",
        "Template\n+ Cross-Tmpl",
    ]
    mean_accs = [
        overall_baseline_mean * 100,
        overall_frozen_mean * 100,
        overall_regenerated_mean * 100,
        overall_patched_mean * 100,
        overall_cross_template_mean_mean * 100,
        overall_template_plain_mean * 100,
        overall_template_patched_mean * 100,
        overall_template_frozen_mean * 100,
        overall_template_cross_template_mean_mean * 100,
    ]
    std_accs = [
        overall_baseline_std * 100,
        overall_frozen_std * 100,
        overall_regenerated_std * 100,
        overall_patched_std * 100,
        overall_cross_template_mean_std * 100,
        overall_template_plain_std * 100,
        overall_template_patched_std * 100,
        overall_template_frozen_std * 100,
        overall_template_cross_template_mean_std * 100,
    ]
    colors = [
        "#2ecc71",
        "#3498db",
        "#e74c3c",
        "#9b59b6",
        "#95a5a6",
        "#f39c12",
        "#16a085",
        "#e67e22",
        "#34495e",
    ]

    bars = ax1.bar(
        methods,
        mean_accs,
        yerr=std_accs,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
        capsize=5,
        error_kw={"linewidth": 2},
    )
    ax1.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Mean Accuracy Across All {num_templates} Templates\n({num_test_cases} Test Cases per Template, Error bars = std dev)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_ylim([0, 100])
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, acc, std in zip(bars, mean_accs, std_accs):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + 2,
            f"{acc:.1f}%\n±{std:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    chart_filename1 = "nov_29_all_conditions_averaged_all_templates.png"
    fig1.savefig(chart_filename1, dpi=300, bbox_inches="tight")
    print(f"Chart 1 saved to {chart_filename1}")

    # Plot 2: Per-template results for all conditions
    fig2, ax2 = plt.subplots(1, 1, figsize=(20, 6))
    template_labels = [f"T{i + 1}" for i in range(num_templates)]

    # Prepare data for grouped bar chart
    x = np.arange(len(template_labels))
    width = 0.09  # Width of bars (adjusted for 9 conditions)

    condition_data = [
        (
            [tr["baseline_mean_acc"] * 100 for tr in all_template_results],
            "Baseline",
            "#2ecc71",
        ),
        (
            [tr["frozen_mean_acc"] * 100 for tr in all_template_results],
            "Mean-Abl+Frozen",
            "#3498db",
        ),
        (
            [tr["regenerated_mean_acc"] * 100 for tr in all_template_results],
            "Mean-Abl+Regen",
            "#e74c3c",
        ),
        (
            [tr["patched_mean_acc"] * 100 for tr in all_template_results],
            "Mean-Abl+Patched",
            "#9b59b6",
        ),
        (
            [tr["cross_template_mean_mean_acc"] * 100 for tr in all_template_results],
            "Mean-Abl+Cross-Tmpl",
            "#95a5a6",
        ),
        (
            [tr["template_plain_mean_acc"] * 100 for tr in all_template_results],
            "Template Plain",
            "#f39c12",
        ),
        (
            [tr["template_patched_mean_acc"] * 100 for tr in all_template_results],
            "Template+Patched",
            "#16a085",
        ),
        (
            [tr["template_frozen_mean_acc"] * 100 for tr in all_template_results],
            "Template+Frozen",
            "#e67e22",
        ),
        (
            [
                tr["template_cross_template_mean_mean_acc"] * 100
                for tr in all_template_results
            ],
            "Template+Cross-Tmpl",
            "#34495e",
        ),
    ]

    for i, (data, label, color) in enumerate(condition_data):
        offset = (i - len(condition_data) / 2) * width + width / 2
        bars = ax2.bar(
            x + offset,
            data,
            width,
            label=label,
            color=color,
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )

    ax2.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Template Index", fontsize=12, fontweight="bold")
    ax2.set_title(
        f"Accuracy Per Template - All Conditions\n({num_test_cases} Test Cases per Template)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(template_labels)
    ax2.set_ylim([0, 100])
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.legend(fontsize=9, loc="upper left", ncol=2)

    plt.tight_layout()
    chart_filename2 = "nov_29_all_conditions_per_template.png"
    fig2.savefig(chart_filename2, dpi=300, bbox_inches="tight")
    print(f"Chart 2 saved to {chart_filename2}")

    # Calculate overall position-wise accuracies for saving
    overall_position_accuracies = []
    overall_position_stds = []
    if num_samples_per_prompt > 1:
        all_position_accuracies = [[] for _ in range(num_samples_per_prompt)]
        for template_result in all_template_results:
            pos_accs = template_result["baseline_position_accuracies"]
            for pos_idx in range(len(pos_accs)):
                if pos_idx < len(all_position_accuracies):
                    all_position_accuracies[pos_idx].append(pos_accs[pos_idx])

        for pos in range(num_samples_per_prompt):
            if len(all_position_accuracies[pos]) > 0:
                overall_position_accuracies.append(
                    float(np.mean(all_position_accuracies[pos]))
                )
                overall_position_stds.append(
                    float(np.std(all_position_accuracies[pos]))
                )
            else:
                overall_position_accuracies.append(0.0)
                overall_position_stds.append(0.0)
    else:
        # Single sample case
        overall_position_accuracies = [float(overall_baseline_mean)]
        overall_position_stds = [float(overall_baseline_std)]

    # Save results
    results_file = "nov_29_multi_template_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "num_templates": num_templates,
                "num_test_cases_per_template": num_test_cases,
                "num_samples_per_prompt": num_samples_per_prompt,
                "temperature": temperature,
                "overall_baseline_mean_accuracy": float(overall_baseline_mean),
                "overall_baseline_std_accuracy": float(overall_baseline_std),
                "overall_frozen_mean_accuracy": float(overall_frozen_mean),
                "overall_frozen_std_accuracy": float(overall_frozen_std),
                "overall_regenerated_mean_accuracy": float(overall_regenerated_mean),
                "overall_regenerated_std_accuracy": float(overall_regenerated_std),
                "overall_patched_mean_accuracy": float(overall_patched_mean),
                "overall_patched_std_accuracy": float(overall_patched_std),
                "overall_cross_template_mean_mean_accuracy": float(
                    overall_cross_template_mean_mean
                ),
                "overall_cross_template_mean_std_accuracy": float(
                    overall_cross_template_mean_std
                ),
                "overall_template_plain_mean_accuracy": float(
                    overall_template_plain_mean
                ),
                "overall_template_plain_std_accuracy": float(
                    overall_template_plain_std
                ),
                "overall_template_patched_mean_accuracy": float(
                    overall_template_patched_mean
                ),
                "overall_template_patched_std_accuracy": float(
                    overall_template_patched_std
                ),
                "overall_template_frozen_mean_accuracy": float(
                    overall_template_frozen_mean
                ),
                "overall_template_frozen_std_accuracy": float(
                    overall_template_frozen_std
                ),
                "overall_template_cross_template_mean_mean_accuracy": float(
                    overall_template_cross_template_mean_mean
                ),
                "overall_template_cross_template_mean_std_accuracy": float(
                    overall_template_cross_template_mean_std
                ),
                "overall_position_accuracies": overall_position_accuracies,
                "overall_position_stds": overall_position_stds,
                "per_template_results": [
                    {
                        "template_idx": tr["template_idx"],
                        "baseline_mean_accuracy": float(tr["baseline_mean_acc"]),
                        "baseline_std_accuracy": float(tr["baseline_std_acc"]),
                        "baseline_position_accuracies": [
                            float(x) for x in tr["baseline_position_accuracies"]
                        ],
                        "frozen_mean_accuracy": float(tr["frozen_mean_acc"]),
                        "frozen_std_accuracy": float(tr["frozen_std_acc"]),
                        "frozen_position_accuracies": [
                            float(x) for x in tr["frozen_position_accuracies"]
                        ],
                        "regenerated_mean_accuracy": float(tr["regenerated_mean_acc"]),
                        "regenerated_std_accuracy": float(tr["regenerated_std_acc"]),
                        "regenerated_position_accuracies": [
                            float(x) for x in tr["regenerated_position_accuracies"]
                        ],
                        "patched_mean_accuracy": float(tr["patched_mean_acc"]),
                        "patched_std_accuracy": float(tr["patched_std_acc"]),
                        "patched_position_accuracies": [
                            float(x) for x in tr["patched_position_accuracies"]
                        ],
                        "cross_template_mean_mean_accuracy": float(
                            tr["cross_template_mean_mean_acc"]
                        ),
                        "cross_template_mean_std_accuracy": float(
                            tr["cross_template_mean_std_acc"]
                        ),
                        "cross_template_mean_position_accuracies": [
                            float(x)
                            for x in tr["cross_template_mean_position_accuracies"]
                        ],
                        "template_plain_mean_accuracy": float(
                            tr["template_plain_mean_acc"]
                        ),
                        "template_plain_std_accuracy": float(
                            tr["template_plain_std_acc"]
                        ),
                        "template_plain_position_accuracies": [
                            float(x) for x in tr["template_plain_position_accuracies"]
                        ],
                        "template_patched_mean_accuracy": float(
                            tr["template_patched_mean_acc"]
                        ),
                        "template_patched_std_accuracy": float(
                            tr["template_patched_std_acc"]
                        ),
                        "template_patched_position_accuracies": [
                            float(x) for x in tr["template_patched_position_accuracies"]
                        ],
                        "template_frozen_mean_accuracy": float(
                            tr["template_frozen_mean_acc"]
                        ),
                        "template_frozen_std_accuracy": float(
                            tr["template_frozen_std_acc"]
                        ),
                        "template_frozen_position_accuracies": [
                            float(x) for x in tr["template_frozen_position_accuracies"]
                        ],
                        "template_cross_template_mean_mean_accuracy": float(
                            tr["template_cross_template_mean_mean_acc"]
                        ),
                        "template_cross_template_mean_std_accuracy": float(
                            tr["template_cross_template_mean_std_acc"]
                        ),
                        "template_cross_template_mean_position_accuracies": [
                            float(x)
                            for x in tr[
                                "template_cross_template_mean_position_accuracies"
                            ]
                        ],
                    }
                    for tr in all_template_results
                ],
                "all_results": all_template_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_file}")

    plt.show()


if __name__ == "__main__":
    main()

# %%
