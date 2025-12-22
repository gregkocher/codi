# %%
# ABOUTME: Step2 latent patching experiment testing if position 4 latent encodes step2 value.
# ABOUTME: Patches latent with mean/random vectors and measures answer frequency shifts.
"""
Step2 Latent Patching Experiment

This experiment tests whether the 5th latent vector (position 4) encodes the second
intermediate value step2 = (X+Y)*Z. By patching this vector with embeddings gathered
from prompts with a different step2 value, we observe if the model's output shifts
toward answers consistent with the patched value.

Key hypothesis: If latent position 4 encodes step2, patching it should cause the model
to output answers consistent with the patched step2 value rather than the original.

Baselines tested:
1. Normal: No patching
2. Patched (latent only): Patch only the latent embedding at the target position
3. Patched (latent + residual): Patch both the latent embedding AND residual stream
   activations at all layers at the target position
4. Patched (cross template): Patch latent from other templates with same step2 value
5. Patched (random): Patch with random vector matching latent statistics (control)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import extract_answer_number
from src.model import CODI
from src.templates import ADDITION_FIRST_TEMPLATES, SUBTRACTION_FIRST_TEMPLATES

# ============================================================================
# Utility Functions
# ============================================================================


def prepare_inputs(model, tokenizer, prompt):
    """Construct input sequence: [Prompt Tokens] + [BOCOT]"""
    device = model.codi.device

    inputs = tokenizer(
        prompt, return_tensors="pt", padding=False, add_special_tokens=True
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    bot_id = tokenizer.convert_tokens_to_ids("<|bocot|>")
    input_ids_bot = torch.cat(
        [input_ids, torch.tensor([[bot_id]], device=device)], dim=1
    )
    attention_mask_bot = torch.cat(
        [attention_mask, torch.ones((1, 1), device=device)], dim=1
    )

    return input_ids_bot, attention_mask_bot


def get_answer(x: int, y: int, z: int, operation: str) -> tuple[int, int, int]:
    """
    Compute ground truth answer and intermediate values.

    Returns: (answer, step_1, step_2)
    """
    if operation == "addition":
        step_1 = x + y
    elif operation == "subtraction":
        step_1 = x - y
    else:
        raise ValueError(f"Unknown operation: {operation}")
    step_2 = step_1 * z
    answer = step_1 + step_2
    return answer, step_1, step_2


def compute_patched_expected_answer(step1: int, step2_gather: int) -> int:
    """Compute what the answer would be if step2 was step2_gather."""
    return step1 + step2_gather


# ============================================================================
# Model Utilities
# ============================================================================


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

    print("Warning: Standard layer path not found. Searching modules...")
    for name, module in model.codi.named_modules():
        if "layers" in name and isinstance(module, torch.nn.ModuleList):
            return module

    raise AttributeError(f"Could not find transformer layers in {type(model.codi)}")


# ============================================================================
# Latent Embedding Collection
# ============================================================================


def collect_latent_embeddings(model, tokenizer, prompt, num_latent_iterations=6):
    """
    Collect all latent embeddings including initial and loop outputs.

    Returns a list of num_latent_iterations + 1 tensors:
    - Index 0: Initial latent embedding from BOCOT position (before loop)
    - Index 1 to num_latent_iterations: Output embeddings from each latent iteration

    Each tensor has shape [1, 1, hidden_size].
    """
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)

    latent_outputs = []

    with torch.no_grad():
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=None,
            attention_mask=attention_mask,
        )
        past_key_values = outputs.past_key_values

        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        if model.use_prj:
            latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        latent_outputs.append(latent_embd.cpu().clone())

        for _ in range(num_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )

            past_key_values = outputs.past_key_values

            output_latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                output_latent_embd = model.prj(output_latent_embd).to(
                    dtype=model.codi.dtype
                )

            latent_outputs.append(output_latent_embd.cpu().clone())
            latent_embd = output_latent_embd

    print(len(latent_outputs))

    return latent_outputs


def compute_mean_latent_at_position(latent_list, position):
    """
    Compute mean latent embedding at a specific position from a list of prompts.

    Args:
        latent_list: List of lists, where latent_list[prompt_idx][position]
                     is the latent embedding for that prompt at that position.
        position: The latent position to average.

    Returns:
        Mean embedding tensor at the specified position.
    """
    embeddings = [prompt_latents[position] for prompt_latents in latent_list]
    stacked = torch.cat(embeddings, dim=0)
    mean_emb = stacked.mean(dim=0, keepdim=True)
    return mean_emb


def collect_latent_residual_streams(model, tokenizer, prompt, num_latent_iterations=6):
    """
    Collect residual stream activations (hidden states at all layers) for each latent position.

    Returns a list of num_latent_iterations + 1 dicts:
    - Index 0: Initial latent position (BOCOT position)
    - Index 1 to num_latent_iterations: Output from each latent iteration

    Each dict contains:
    - 'latent_embd': The latent embedding at this position [1, 1, hidden_size]
    - 'residual_streams': Tensor of shape [num_layers, 1, hidden_size] with residual
                          stream activations at the last position for each layer
    """
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)
    layers = get_transformer_layers(model)
    num_layers = len(layers)

    latent_outputs = []

    with torch.no_grad():
        # Initial forward pass to get first latent
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=None,
            attention_mask=attention_mask,
        )
        past_key_values = outputs.past_key_values

        # Collect residual streams at the last position (BOCOT) for initial
        # hidden_states: tuple of (num_layers + 1) tensors of shape [batch, seq, hidden]
        # We skip layer 0 (embedding layer) and take layers 1 to num_layers
        initial_residuals = torch.stack(
            [h[:, -1, :] for h in outputs.hidden_states[1 : num_layers + 1]], dim=0
        )

        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        if model.use_prj:
            latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        latent_outputs.append(
            {
                "latent_embd": latent_embd.cpu().clone(),
                "residual_streams": initial_residuals.cpu().clone(),
            }
        )

        # Latent iterations
        for _ in range(num_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )

            past_key_values = outputs.past_key_values

            # Collect residual streams at the last position for this iteration
            iter_residuals = torch.stack(
                [h[:, -1, :] for h in outputs.hidden_states[1 : num_layers + 1]], dim=0
            )

            output_latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                output_latent_embd = model.prj(output_latent_embd).to(
                    dtype=model.codi.dtype
                )

            latent_outputs.append(
                {
                    "latent_embd": output_latent_embd.cpu().clone(),
                    "residual_streams": iter_residuals.cpu().clone(),
                }
            )
            latent_embd = output_latent_embd

    return latent_outputs


def compute_mean_residual_at_position(residual_list, position):
    """
    Compute mean residual stream activations at a specific position from a list of prompts.

    Args:
        residual_list: List of lists, where residual_list[prompt_idx][position]
                       is a dict with 'residual_streams' tensor of shape [num_layers, 1, hidden_size]
        position: The latent position to average.

    Returns:
        Mean residual streams tensor of shape [num_layers, 1, hidden_size].
    """
    residuals = [
        prompt_residuals[position]["residual_streams"]
        for prompt_residuals in residual_list
    ]
    # Each residual is [num_layers, 1, hidden_size], stack along new batch dim
    stacked = torch.stack(residuals, dim=0)  # [num_prompts, num_layers, 1, hidden_size]
    mean_residual = stacked.mean(dim=0)  # [num_layers, 1, hidden_size]
    return mean_residual


def compute_latent_statistics(all_latents, position):
    """
    Compute mean and std of latent embeddings at a specific position across all prompts.

    Args:
        all_latents: Dict mapping prompt_idx -> list of latent embeddings per position
        position: The latent position to compute statistics for

    Returns:
        Tuple of (global_mean, global_std) tensors with shape [1, 1, hidden_size]
    """
    embeddings = [latents[position] for latents in all_latents.values()]
    stacked = torch.cat(embeddings, dim=0)  # [num_prompts, 1, hidden_size]
    global_mean = stacked.mean(dim=0, keepdim=True)  # [1, 1, hidden_size]
    global_std = stacked.std(dim=0, keepdim=True)  # [1, 1, hidden_size]
    return global_mean, global_std


def generate_random_latent(global_mean, global_std, seed=None):
    """
    Generate a random latent vector with similar statistics to real latents.

    Args:
        global_mean: Mean of real latents [1, 1, hidden_size]
        global_std: Std of real latents [1, 1, hidden_size]
        seed: Optional random seed for reproducibility

    Returns:
        Random latent tensor with shape [1, 1, hidden_size]
    """
    if seed is not None:
        torch.manual_seed(seed)
    random_latent = torch.randn_like(global_mean) * global_std + global_mean
    return random_latent


# ============================================================================
# Generation with Mean Latent Patching
# ============================================================================


def generate_with_mean_latent_patching(
    model,
    tokenizer,
    prompt,
    patch_position,
    mean_latent,
    max_new_tokens=128,
    num_latent_iterations=6,
    temperature=0.1,
    greedy=True,
):
    """
    Generate with the latent embedding patched at a specific position.

    Args:
        patch_position: Which latent vector to patch (0 = initial, 1-6 = loop outputs).
        mean_latent: The mean latent embedding to use for patching.
    """
    device = model.codi.device
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)
    batch_size = input_ids.size(0)

    with torch.no_grad():
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=None,
            attention_mask=attention_mask,
        )
        past_key_values = outputs.past_key_values

        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        if model.use_prj:
            latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        if patch_position == 0:
            latent_embd = mean_latent.to(device).to(dtype=model.codi.dtype)

        for i in range(num_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )

            past_key_values = outputs.past_key_values

            output_latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                output_latent_embd = model.prj(output_latent_embd).to(
                    dtype=model.codi.dtype
                )

            if patch_position == i + 1:
                latent_embd = mean_latent.to(device).to(dtype=model.codi.dtype)
            else:
                latent_embd = output_latent_embd

        eot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")
        eot_ids = torch.tensor([[eot_id]], dtype=torch.long, device=device)
        eot_emb = model.get_embd(model.codi, model.model_name)(eot_ids).to(device)
        output = eot_emb

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        sequences = input_ids.clone()

        for _ in range(max_new_tokens):
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


def generate_with_mean_latent_and_residual_patching(
    model,
    tokenizer,
    prompt,
    patch_position,
    mean_latent,
    mean_residual,
    max_new_tokens=128,
    num_latent_iterations=6,
    temperature=0.1,
    greedy=True,
):
    """
    Generate with both the latent embedding AND residual stream activations patched
    at a specific position.

    Args:
        patch_position: Which latent vector to patch (0 = initial, 1-6 = loop outputs).
        mean_latent: The mean latent embedding to use for patching.
        mean_residual: The mean residual stream activations tensor [num_layers, 1, hidden_size].
    """
    device = model.codi.device
    layers = get_transformer_layers(model)
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)
    batch_size = input_ids.size(0)

    # Track current latent iteration for the hook
    current_latent_iter = [0]  # Use list to allow mutation in nested function
    is_latent_forward = [False]  # Track if we're in a latent forward pass

    def _create_residual_patch_hook(layer_idx):
        """Create a hook that patches the residual stream at the target position."""

        def hook(module, args, output):
            # Only patch during latent iterations at the target position
            if not is_latent_forward[0]:
                return output
            if current_latent_iter[0] != patch_position:
                return output

            if isinstance(output, tuple):
                hidden_states = output[0]
                new_hidden = hidden_states.clone()
                # Patch the last position with the mean residual
                new_hidden[:, -1, :] = (
                    mean_residual[layer_idx, :, :].to(device).to(hidden_states.dtype)
                )
                return (new_hidden,) + output[1:]
            else:
                new_hidden = output.clone()
                new_hidden[:, -1, :] = (
                    mean_residual[layer_idx, :, :].to(device).to(output.dtype)
                )
                return new_hidden

        return hook

    # Register hooks for all layers
    patch_handles = []
    for layer_idx, layer in enumerate(layers):
        hook = _create_residual_patch_hook(layer_idx)
        patch_handles.append(layer.register_forward_hook(hook))

    try:
        with torch.no_grad():
            # Initial forward pass (position 0)
            is_latent_forward[0] = True
            current_latent_iter[0] = 0

            outputs = model.codi(
                input_ids=input_ids,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=None,
                attention_mask=attention_mask,
            )
            past_key_values = outputs.past_key_values

            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

            # Patch latent embedding if position 0
            if patch_position == 0:
                latent_embd = mean_latent.to(device).to(dtype=model.codi.dtype)

            # Latent iterations
            for i in range(num_latent_iterations):
                current_latent_iter[0] = i + 1

                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values,
                )

                past_key_values = outputs.past_key_values

                output_latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if model.use_prj:
                    output_latent_embd = model.prj(output_latent_embd).to(
                        dtype=model.codi.dtype
                    )

                # Patch latent embedding if at target position
                if patch_position == i + 1:
                    latent_embd = mean_latent.to(device).to(dtype=model.codi.dtype)
                else:
                    latent_embd = output_latent_embd

            # Disable hooks for generation phase
            is_latent_forward[0] = False

            eot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")
            eot_ids = torch.tensor([[eot_id]], dtype=torch.long, device=device)
            eot_emb = model.get_embd(model.codi, model.model_name)(eot_ids).to(device)
            output = eot_emb

            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            sequences = input_ids.clone()

            for _ in range(max_new_tokens):
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
    finally:
        # Clean up hooks
        for h in patch_handles:
            h.remove()


def generate_normal(
    model, tokenizer, prompt, num_latent_iterations=6, greedy=True, temperature=1.0
):
    """Generate without patching (normal inference)."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.codi.device)
    attention_mask = tokenizer(prompt, return_tensors="pt").attention_mask.to(
        model.codi.device
    )
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=128,
        num_latent_iterations=num_latent_iterations,
        temperature=temperature,
        greedy=greedy,
        return_latent_vectors=False,
        remove_eos=False,
        output_attentions=False,
        skip_thinking=False,
        output_hidden_states=False,
        sot_token=tokenizer.convert_tokens_to_ids("<|bocot|>"),
        eot_token=tokenizer.convert_tokens_to_ids("<|eocot|>"),
        verbalize_cot=False,
    )
    return output


# ============================================================================
# Prompt Generation and Grouping
# ============================================================================


def find_common_step2_values(num_step2_values, seed=42, min_step2=1, max_step2=10):
    """
    Find step2 values that are common to both addition and subtraction operations.

    Args:
        num_step2_values: Number of step2 values to select
        seed: Random seed
        min_step2: Minimum step2 value (inclusive)
        max_step2: Maximum step2 value (inclusive)

    Returns:
        List of selected common step2 values
    """
    all_combinations = [
        (x, y, z)
        for x in range(1, 20)
        for y in range(1, 20)
        for z in range(1, 20)
        if x > y  # Ensure valid subtraction
    ]

    # Find step2 values for each operation
    addition_step2 = set()
    subtraction_step2 = set()

    for x, y, z in all_combinations:
        _, _, step2_add = get_answer(x, y, z, "addition")
        _, _, step2_sub = get_answer(x, y, z, "subtraction")
        addition_step2.add(step2_add)
        subtraction_step2.add(step2_sub)

    # Find common step2 values within the specified range
    common_step2 = sorted(
        v for v in (addition_step2 & subtraction_step2) if min_step2 <= v <= max_step2
    )
    print(f"Common step2 values in [{min_step2}, {max_step2}]: {common_step2}")

    # Select from common values
    np.random.seed(seed)
    if num_step2_values >= len(common_step2):
        selected_step2_values = common_step2
    else:
        selected_step2_values = list(
            np.random.choice(common_step2, size=num_step2_values, replace=False)
        )
        selected_step2_values = sorted([int(v) for v in selected_step2_values])

    print(f"Selected step2 values: {selected_step2_values}")
    return selected_step2_values


def generate_prompts_for_template(
    template, template_idx, operation, selected_step2_values
):
    """
    Generate prompts for a single template using pre-selected step2 values.

    Args:
        template: The prompt template string
        template_idx: Index of the template
        operation: "addition" or "subtraction"
        selected_step2_values: List of step2 values to use (pre-selected)

    Returns:
        prompts: List of prompt info dicts
        step2_groups: Dict mapping step2 value to list of prompt indices
    """
    # Compute all combinations and group by step2
    all_combinations = [
        (x, y, z)
        for x in range(1, 10)
        for y in range(1, 10)
        for z in range(1, 10)
        if x > y  # Ensure valid subtraction
    ]

    # Group combinations by their step2 value
    step2_to_combinations = defaultdict(list)
    for x, y, z in all_combinations:
        _, step_1, step_2 = get_answer(x, y, z, operation)
        step2_to_combinations[step_2].append((x, y, z, step_1))

    # Generate prompts only for selected step2 values that exist for this operation
    prompts = []
    step2_groups = defaultdict(list)
    idx = 0

    for step2_val in selected_step2_values:
        if step2_val not in step2_to_combinations:
            continue  # This step2 value doesn't exist for this operation
        for x, y, z, step_1 in step2_to_combinations[step2_val]:
            answer = step_1 + step2_val

            prompt_info = {
                "idx": idx,
                "operation": operation,
                "template_idx": template_idx,
                "X": x,
                "Y": y,
                "Z": z,
                "prompt": template.format(X=x, Y=y, Z=z),
                "ground_truth": answer,
                "step_1": step_1,
                "step_2": step2_val,
            }
            prompts.append(prompt_info)
            step2_groups[step2_val].append(idx)
            idx += 1

    return prompts, step2_groups


def load_test_prompts(prompts_json_path, selected_step2_values, limit_templates=None):
    """
    Load test prompts from external JSON file.

    Args:
        prompts_json_path: Path to prompts.json file
        selected_step2_values: List of step2 values to filter by
        limit_templates: Limit number of templates (optional)

    Returns:
        Dict mapping (operation, template_idx) -> list of test prompt dicts
    """
    with open(prompts_json_path) as f:
        data = json.load(f)

    test_prompts = defaultdict(list)

    for entry in data["prompts"]:
        template_idx = entry["template_idx"]

        if limit_templates is not None and template_idx >= limit_templates:
            continue

        x, y, z = entry["X"], entry["Y"], entry["Z"]

        # Process both addition and subtraction
        for operation in ["addition", "subtraction"]:
            _, step1, step2 = get_answer(x, y, z, operation)

            prompt_info = {
                "X": x,
                "Y": y,
                "Z": z,
                "prompt": entry[operation]["prompt"],
                "ground_truth": entry[operation]["ground_truth"],
                "step_1": step1,
                "step_2": step2,
                "template_idx": template_idx,
                "operation": operation,
            }

            test_prompts[(operation, template_idx)].append(prompt_info)

    print(f"Loaded test prompts from {prompts_json_path}")
    for key, prompts in sorted(test_prompts.items()):
        print(f"  {key[0]} template {key[1]}: {len(prompts)} prompts")

    return test_prompts


# ============================================================================
# Main Experiment
# ============================================================================


def main(
    limit_templates: int | None = None,
    limit_test_prompts: int | None = None,
    step2_values: list[int] | None = None,
    patch_position: int = 4,
    greedy: bool = False,
    temperature: float = 1.0,
    num_samples: int = 3,
    seed: int = 42,
    test_prompts_path: str = "prompts/prompts.json",
):
    """
    Run the step2 latent patching experiment.

    Tests four scenarios:
    1. Normal inference (no patching)
    2. Patched with same template (mean latent from same template, different step2)
    3. Patched with cross template (mean latent from OTHER templates, same step2)
    4. Patched latent + residual stream (mean latent AND mean residual stream
       activations at all layers, from same template, different step2)

    Args:
        limit_templates: Limit number of templates to test (for fast testing)
        limit_test_prompts: Limit number of test prompts per template (for fast testing)
        step2_values: List of step2 values to gather latents for (e.g., [6, 10, 12])
        patch_position: Latent position to patch (default 4 = 5th vector)
        greedy: Use greedy decoding
        temperature: Sampling temperature (used when greedy=False)
        num_samples: Number of samples per prompt per condition
        test_prompts_path: Path to external test prompts JSON file
        seed: Random seed
    """
    load_dotenv()
    np.random.seed(seed)

    results_dir = Path(__file__).parent.parent / "results" / "step2_latent_patching"
    results_dir.mkdir(parents=True, exist_ok=True)

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
        remove_eos=False,
        full_precision=True,
    )
    tokenizer = model.tokenizer
    num_latent_iterations = 6

    all_templates = [
        ("addition", ADDITION_FIRST_TEMPLATES),
        ("subtraction", SUBTRACTION_FIRST_TEMPLATES),
    ]

    if limit_templates is not None:
        limited_templates = []
        for operation, templates in all_templates:
            limited_templates.append((operation, templates[:limit_templates]))
        all_templates = limited_templates

    # =========================================================================
    # Phase 1: Generate prompts and collect latents for ALL templates
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: Collecting latent embeddings from all templates")
    print("=" * 80)

    # Use provided step2 values or default
    if step2_values is None:
        step2_values = [10]  # Default values
    selected_step2_values = sorted(step2_values)
    print(f"\nUsing step2 values: {selected_step2_values}")
    # Structure: {template_key: {prompts, step2_groups, selected_step2, latents}}
    # template_key = (operation, template_idx) - treats all templates jointly
    all_template_data = {}

    for operation, templates in all_templates:
        for template_idx, template in enumerate(templates):
            template_key = (operation, template_idx)
            print(f"\nCollecting from {operation} template {template_idx}...")

            prompts, step2_groups = generate_prompts_for_template(
                template, template_idx, operation, selected_step2_values
            )
            print(prompts)
            print(f"  {len(prompts)} prompts, {len(step2_groups)} step2 values")

            # Collect latent embeddings and residual streams
            prompt_latents = {}
            prompt_residuals = {}
            for prompt_info in tqdm(prompts, desc="  Collecting latents & residuals"):
                latent_outputs = collect_latent_embeddings(
                    model,
                    tokenizer,
                    prompt_info["prompt"],
                    num_latent_iterations=num_latent_iterations,
                )
                prompt_latents[prompt_info["idx"]] = latent_outputs

                residual_outputs = collect_latent_residual_streams(
                    model,
                    tokenizer,
                    prompt_info["prompt"],
                    num_latent_iterations=num_latent_iterations,
                )
                prompt_residuals[prompt_info["idx"]] = residual_outputs

            all_template_data[template_key] = {
                "operation": operation,
                "template_idx": template_idx,
                "template": template,
                "prompts": prompts,
                "step2_groups": step2_groups,
                "selected_step2_values": selected_step2_values,
                "prompt_latents": prompt_latents,
                "prompt_residuals": prompt_residuals,
            }

    # =========================================================================
    # Phase 1b: Compute global latent statistics for random baseline
    # =========================================================================
    print("\n" + "-" * 40)
    print("Computing global latent statistics for random baseline...")

    # Collect all latents across all templates
    all_latents_global = {}
    global_idx = 0
    for template_key, data in all_template_data.items():
        for idx, latents in data["prompt_latents"].items():
            all_latents_global[global_idx] = latents
            global_idx += 1

    global_mean, global_std = compute_latent_statistics(
        all_latents_global, patch_position
    )
    print(f"  Global latent stats computed from {len(all_latents_global)} prompts")
    print(
        f"  Mean norm: {global_mean.norm().item():.4f}, Std mean: {global_std.mean().item():.4f}"
    )

    # =========================================================================
    # Phase 2: Compute mean latents (same-template and cross-template)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: Computing mean latents (same-template and cross-template)")
    print("=" * 80)

    for template_key, data in all_template_data.items():
        operation, template_idx = template_key
        selected_step2_values = data["selected_step2_values"]
        step2_groups = data["step2_groups"]
        prompt_latents = data["prompt_latents"]
        prompt_residuals = data["prompt_residuals"]

        # Same-template mean latents
        same_template_means = {}
        for step2_val in selected_step2_values:
            indices = step2_groups[step2_val]
            latent_list = [prompt_latents[idx] for idx in indices]
            same_template_means[step2_val] = compute_mean_latent_at_position(
                latent_list, patch_position
            )
        data["same_template_means"] = same_template_means

        # Same-template mean residual streams
        same_template_residual_means = {}
        for step2_val in selected_step2_values:
            indices = step2_groups[step2_val]
            residual_list = [prompt_residuals[idx] for idx in indices]
            same_template_residual_means[step2_val] = compute_mean_residual_at_position(
                residual_list, patch_position
            )
        data["same_template_residual_means"] = same_template_residual_means

        # Cross-template mean latents (from ALL other templates, all operations)
        cross_template_means = {}
        cross_template_sources = {}
        for step2_val in selected_step2_values:
            cross_latent_list = []
            contributing_templates = []
            for other_key, other_data in all_template_data.items():
                if other_key == template_key:
                    continue  # Skip same template
                if step2_val in other_data["step2_groups"]:
                    other_indices = other_data["step2_groups"][step2_val]
                    for idx in other_indices:
                        cross_latent_list.append(other_data["prompt_latents"][idx])
                    contributing_templates.append(
                        f"{other_data['operation']}_{other_data['template_idx']}"
                    )

            if cross_latent_list:
                cross_template_means[step2_val] = compute_mean_latent_at_position(
                    cross_latent_list, patch_position
                )
                cross_template_sources[step2_val] = {
                    "templates": contributing_templates,
                    "num_latents": len(cross_latent_list),
                }
            else:
                cross_template_means[step2_val] = None
                cross_template_sources[step2_val] = None

        data["cross_template_means"] = cross_template_means
        data["cross_template_sources"] = cross_template_sources

        # Print detailed info about cross-template contributions
        print(
            f"{operation} template {template_idx}: "
            f"{len(same_template_means)} same-template means, "
            f"{sum(1 for v in cross_template_means.values() if v is not None)} cross-template means"
        )
        for s2 in selected_step2_values:
            src = cross_template_sources.get(s2)
            info = (
                f"{src['num_latents']} from {', '.join(src['templates'])}"
                if src
                else "none"
            )
            print(f"    step2={s2}: cross=[{info}]")

    # =========================================================================
    # Phase 3: Run patching experiment on external test prompts
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: Running patching experiment on external test prompts")
    print("=" * 80)

    # Load external test prompts
    test_prompts_by_template = load_test_prompts(
        test_prompts_path, selected_step2_values, limit_templates
    )

    all_results = []
    conditions = [
        "normal",
        "patched_same",
        "patched_cross",
        "patched_latent_and_resid",
        "patched_random",
    ]
    aggregate_counts = {
        cond: {"original_correct": 0, "patched_expected": 0, "other": 0}
        for cond in conditions
    }

    # Per-operation counts
    operation_counts = {
        op: {
            cond: {"original_correct": 0, "patched_expected": 0, "other": 0}
            for cond in conditions
        }
        for op in ["addition", "subtraction"]
    }

    def categorize_answer(answer, original, patched_exp):
        if answer == original:
            return "original_correct"
        elif answer == patched_exp:
            return "patched_expected"
        else:
            return "other"

    for template_key, data in all_template_data.items():
        operation, template_idx = template_key
        print(f"\nTesting {operation} template {template_idx}...")

        template = data["template"]
        # Use external test prompts instead of prompts used for mean calculation
        test_prompts = test_prompts_by_template.get((operation, template_idx), [])
        selected_step2_values = data["selected_step2_values"]
        same_template_means = data["same_template_means"]
        same_template_residual_means = data["same_template_residual_means"]
        cross_template_means = data["cross_template_means"]

        template_results = {
            "operation": operation,
            "template_idx": template_idx,
            "template": template,
            "patch_position": patch_position,
            "test_results": [],
        }

        template_counts = {
            cond: {"original_correct": 0, "patched_expected": 0, "other": 0}
            for cond in conditions
        }

        if not test_prompts:
            print("  No test prompts available for this template")
            continue

        for step2_gather in tqdm(selected_step2_values, desc="Testing step2 values"):
            mean_latent_same = same_template_means.get(step2_gather)
            mean_residual_same = same_template_residual_means.get(step2_gather)
            mean_latent_cross = cross_template_means.get(step2_gather)

            if mean_latent_same is None:
                print(f"    step2_gather={step2_gather}: No mean latent available")
                continue

            # Filter test prompts where step2 != step2_gather
            filtered_test_prompts = [
                p for p in test_prompts if p["step_2"] != step2_gather
            ]

            # Apply limit if specified
            if limit_test_prompts is not None:
                filtered_test_prompts = filtered_test_prompts[:limit_test_prompts]

            if not filtered_test_prompts:
                print(f"    step2_gather={step2_gather}: No test prompts found")
                continue

            for test_idx, prompt_info in enumerate(filtered_test_prompts):
                prompt = prompt_info["prompt"]
                original_answer = prompt_info["ground_truth"]
                step1 = prompt_info["step_1"]
                patched_expected = compute_patched_expected_answer(step1, step2_gather)

                sample_results = {
                    "prompt_idx": test_idx,
                    "X": prompt_info["X"],
                    "Y": prompt_info["Y"],
                    "Z": prompt_info["Z"],
                    "original_answer": original_answer,
                    "step1": step1,
                    "step2_original": prompt_info["step_2"],
                    "step2_gather": step2_gather,
                    "patched_expected": patched_expected,
                    "has_cross": mean_latent_cross is not None,
                    "samples": [],
                }

                for sample_idx in range(num_samples):
                    # Generate a fresh random latent for each sample
                    random_latent = generate_random_latent(
                        global_mean,
                        global_std,
                        seed=seed + sample_idx + test_idx * 1000,
                    )

                    # Normal inference
                    output_normal = generate_normal(
                        model,
                        tokenizer,
                        prompt,
                        num_latent_iterations=num_latent_iterations,
                        greedy=greedy,
                        temperature=temperature,
                    )
                    text_normal = tokenizer.decode(
                        output_normal["sequences"][0], skip_special_tokens=False
                    )
                    answer_normal = extract_answer_number(text_normal)
                    if answer_normal is not None and answer_normal != float("inf"):
                        answer_normal = int(answer_normal)

                    # Patched with same template
                    output_patched_same = generate_with_mean_latent_patching(
                        model,
                        tokenizer,
                        prompt,
                        patch_position=patch_position,
                        mean_latent=mean_latent_same,
                        num_latent_iterations=num_latent_iterations,
                        greedy=greedy,
                        temperature=temperature,
                    )
                    text_patched_same = tokenizer.decode(
                        output_patched_same["sequences"][0],
                        skip_special_tokens=False,
                    )
                    answer_patched_same = extract_answer_number(text_patched_same)
                    if answer_patched_same is not None and answer_patched_same != float(
                        "inf"
                    ):
                        answer_patched_same = int(answer_patched_same)

                    # Patched with cross template (if available)
                    answer_patched_cross = None
                    cat_patched_cross = None
                    if mean_latent_cross is not None:
                        output = generate_with_mean_latent_patching(
                            model,
                            tokenizer,
                            prompt,
                            patch_position=patch_position,
                            mean_latent=mean_latent_cross,
                            num_latent_iterations=num_latent_iterations,
                            greedy=greedy,
                            temperature=temperature,
                        )
                        text = tokenizer.decode(
                            output["sequences"][0], skip_special_tokens=False
                        )
                        answer_patched_cross = extract_answer_number(text)
                        if (
                            answer_patched_cross is not None
                            and answer_patched_cross != float("inf")
                        ):
                            answer_patched_cross = int(answer_patched_cross)

                    # Patched with both latent embedding AND residual stream
                    output_patched_latent_and_resid = (
                        generate_with_mean_latent_and_residual_patching(
                            model,
                            tokenizer,
                            prompt,
                            patch_position=patch_position,
                            mean_latent=mean_latent_same,
                            mean_residual=mean_residual_same,
                            num_latent_iterations=num_latent_iterations,
                            greedy=greedy,
                            temperature=temperature,
                        )
                    )
                    text_patched_latent_and_resid = tokenizer.decode(
                        output_patched_latent_and_resid["sequences"][0],
                        skip_special_tokens=False,
                    )
                    answer_patched_latent_and_resid = extract_answer_number(
                        text_patched_latent_and_resid
                    )
                    if (
                        answer_patched_latent_and_resid is not None
                        and answer_patched_latent_and_resid != float("inf")
                    ):
                        answer_patched_latent_and_resid = int(
                            answer_patched_latent_and_resid
                        )

                    # Patched with random vector (control baseline)
                    output_patched_random = generate_with_mean_latent_patching(
                        model,
                        tokenizer,
                        prompt,
                        patch_position=patch_position,
                        mean_latent=random_latent,
                        num_latent_iterations=num_latent_iterations,
                        greedy=greedy,
                        temperature=temperature,
                    )
                    text_patched_random = tokenizer.decode(
                        output_patched_random["sequences"][0],
                        skip_special_tokens=False,
                    )
                    answer_patched_random = extract_answer_number(text_patched_random)
                    if (
                        answer_patched_random is not None
                        and answer_patched_random != float("inf")
                    ):
                        answer_patched_random = int(answer_patched_random)

                    # Categorize answers
                    cat_normal = categorize_answer(
                        answer_normal, original_answer, patched_expected
                    )
                    cat_patched_same = categorize_answer(
                        answer_patched_same, original_answer, patched_expected
                    )
                    if answer_patched_cross is not None:
                        cat_patched_cross = categorize_answer(
                            answer_patched_cross, original_answer, patched_expected
                        )
                    cat_patched_latent_and_resid = categorize_answer(
                        answer_patched_latent_and_resid,
                        original_answer,
                        patched_expected,
                    )
                    cat_patched_random = categorize_answer(
                        answer_patched_random, original_answer, patched_expected
                    )

                    # Update counts
                    template_counts["normal"][cat_normal] += 1
                    template_counts["patched_same"][cat_patched_same] += 1
                    template_counts["patched_latent_and_resid"][
                        cat_patched_latent_and_resid
                    ] += 1
                    template_counts["patched_random"][cat_patched_random] += 1
                    aggregate_counts["normal"][cat_normal] += 1
                    aggregate_counts["patched_same"][cat_patched_same] += 1
                    aggregate_counts["patched_latent_and_resid"][
                        cat_patched_latent_and_resid
                    ] += 1
                    aggregate_counts["patched_random"][cat_patched_random] += 1
                    operation_counts[operation]["normal"][cat_normal] += 1
                    operation_counts[operation]["patched_same"][cat_patched_same] += 1
                    operation_counts[operation]["patched_latent_and_resid"][
                        cat_patched_latent_and_resid
                    ] += 1
                    operation_counts[operation]["patched_random"][
                        cat_patched_random
                    ] += 1

                    if cat_patched_cross is not None:
                        template_counts["patched_cross"][cat_patched_cross] += 1
                        aggregate_counts["patched_cross"][cat_patched_cross] += 1
                        operation_counts[operation]["patched_cross"][
                            cat_patched_cross
                        ] += 1

                    sample_results["samples"].append(
                        {
                            "sample_idx": sample_idx,
                            "answer_normal": answer_normal,
                            "answer_patched_same": answer_patched_same,
                            "answer_patched_cross": answer_patched_cross,
                            "answer_patched_latent_and_resid": answer_patched_latent_and_resid,
                            "answer_patched_random": answer_patched_random,
                            "category_normal": cat_normal,
                            "category_patched_same": cat_patched_same,
                            "category_patched_cross": cat_patched_cross,
                            "category_patched_latent_and_resid": cat_patched_latent_and_resid,
                            "category_patched_random": cat_patched_random,
                        }
                    )

                template_results["test_results"].append(sample_results)

        template_results["counts"] = template_counts
        all_results.append(template_results)

        print(f"  Normal: {template_counts['normal']}")
        print(f"  Patched (same template): {template_counts['patched_same']}")
        print(f"  Patched (cross template): {template_counts['patched_cross']}")
        print(
            f"  Patched (latent+resid): {template_counts['patched_latent_and_resid']}"
        )
        print(f"  Patched (random): {template_counts['patched_random']}")

    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    print(f"Normal inference:           {aggregate_counts['normal']}")
    print(f"Patched (same template):    {aggregate_counts['patched_same']}")
    print(f"Patched (cross template):   {aggregate_counts['patched_cross']}")
    print(f"Patched (latent+resid):     {aggregate_counts['patched_latent_and_resid']}")
    print(f"Patched (random):           {aggregate_counts['patched_random']}")

    print("\n" + "-" * 40)
    print("RESULTS BY OPERATION")
    print("-" * 40)
    for op in ["addition", "subtraction"]:
        print(f"\n{op.upper()}:")
        print(f"  Normal:             {operation_counts[op]['normal']}")
        print(f"  Patched (same):     {operation_counts[op]['patched_same']}")
        print(f"  Patched (cross):    {operation_counts[op]['patched_cross']}")
        print(
            f"  Patched (lat+res):  {operation_counts[op]['patched_latent_and_resid']}"
        )
        print(f"  Patched (random):   {operation_counts[op]['patched_random']}")

    output_data = {
        "config": {
            "limit_templates": limit_templates,
            "limit_test_prompts": limit_test_prompts,
            "step2_values": selected_step2_values,
            "patch_position": patch_position,
            "greedy": greedy,
            "temperature": temperature,
            "num_samples": num_samples,
            "seed": seed,
        },
        "aggregate_counts": aggregate_counts,
        "operation_counts": operation_counts,
        "template_results": all_results,
    }

    output_file = results_dir / "results.json"
    print(f"\nSaving results to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print("\nCreating visualization...")
    create_visualizations(
        aggregate_counts, operation_counts, all_results, results_dir, num_samples
    )

    print("\nExperiment complete!")


def compute_per_prompt_stats(all_results, num_samples):
    """
    Compute per-prompt frequency statistics for each category.

    For each prompt, computes the frequency of each category across its samples,
    then returns mean and std across all prompts.

    Returns:
        dict with 'normal', 'patched_same', 'patched_cross', 'patched_latent_and_resid',
        'patched_random' keys, each containing:
            - 'mean': dict of category -> mean frequency
            - 'std': dict of category -> std frequency
            - 'sem': dict of category -> standard error of mean
    """
    categories = ["original_correct", "patched_expected", "other"]
    stat_conditions = [
        "normal",
        "patched_same",
        "patched_cross",
        "patched_latent_and_resid",
        "patched_random",
    ]

    # Collect per-prompt frequencies
    per_prompt = {cond: {c: [] for c in categories} for cond in stat_conditions}

    for result in all_results:
        for test_result in result["test_results"]:
            samples = test_result["samples"]
            if not samples:
                continue

            # Count categories for this prompt's samples
            counts = {cond: {c: 0 for c in categories} for cond in stat_conditions}
            cross_count = 0

            for sample in samples:
                counts["normal"][sample["category_normal"]] += 1
                counts["patched_same"][sample["category_patched_same"]] += 1
                counts["patched_latent_and_resid"][
                    sample["category_patched_latent_and_resid"]
                ] += 1
                counts["patched_random"][sample["category_patched_random"]] += 1
                if sample["category_patched_cross"] is not None:
                    counts["patched_cross"][sample["category_patched_cross"]] += 1
                    cross_count += 1

            # Convert to frequencies for this prompt
            n_samples = len(samples)
            for c in categories:
                per_prompt["normal"][c].append(counts["normal"][c] / n_samples)
                per_prompt["patched_same"][c].append(
                    counts["patched_same"][c] / n_samples
                )
                per_prompt["patched_latent_and_resid"][c].append(
                    counts["patched_latent_and_resid"][c] / n_samples
                )
                per_prompt["patched_random"][c].append(
                    counts["patched_random"][c] / n_samples
                )
                if cross_count > 0:
                    per_prompt["patched_cross"][c].append(
                        counts["patched_cross"][c] / cross_count
                    )

    # Compute mean and std across prompts
    stats = {cond: {"mean": {}, "std": {}, "sem": {}} for cond in stat_conditions}

    for cond in stat_conditions:
        for c in categories:
            if per_prompt[cond][c]:
                stats[cond]["mean"][c] = np.mean(per_prompt[cond][c])
                stats[cond]["std"][c] = np.std(per_prompt[cond][c])
                stats[cond]["sem"][c] = np.std(per_prompt[cond][c]) / np.sqrt(
                    len(per_prompt[cond][c])
                )
            else:
                stats[cond]["mean"][c] = 0.0
                stats[cond]["std"][c] = 0.0
                stats[cond]["sem"][c] = 0.0

    return stats


def compute_template_stats(test_results):
    """Compute per-prompt statistics for a single template."""
    categories = ["original_correct", "patched_expected", "other"]
    stat_conditions = [
        "normal",
        "patched_same",
        "patched_cross",
        "patched_latent_and_resid",
        "patched_random",
    ]

    per_prompt = {cond: {c: [] for c in categories} for cond in stat_conditions}

    for test_result in test_results:
        samples = test_result["samples"]
        if not samples:
            continue

        counts = {cond: {c: 0 for c in categories} for cond in stat_conditions}
        cross_count = 0

        for sample in samples:
            counts["normal"][sample["category_normal"]] += 1
            counts["patched_same"][sample["category_patched_same"]] += 1
            counts["patched_latent_and_resid"][
                sample["category_patched_latent_and_resid"]
            ] += 1
            counts["patched_random"][sample["category_patched_random"]] += 1
            if sample["category_patched_cross"] is not None:
                counts["patched_cross"][sample["category_patched_cross"]] += 1
                cross_count += 1

        n_samples = len(samples)
        for c in categories:
            per_prompt["normal"][c].append(counts["normal"][c] / n_samples)
            per_prompt["patched_same"][c].append(counts["patched_same"][c] / n_samples)
            per_prompt["patched_latent_and_resid"][c].append(
                counts["patched_latent_and_resid"][c] / n_samples
            )
            per_prompt["patched_random"][c].append(
                counts["patched_random"][c] / n_samples
            )
            if cross_count > 0:
                per_prompt["patched_cross"][c].append(
                    counts["patched_cross"][c] / cross_count
                )

    stats = {cond: {"mean": {}, "std": {}, "sem": {}} for cond in stat_conditions}

    for cond in stat_conditions:
        for c in categories:
            if per_prompt[cond][c]:
                stats[cond]["mean"][c] = np.mean(per_prompt[cond][c])
                stats[cond]["std"][c] = np.std(per_prompt[cond][c])
                stats[cond]["sem"][c] = np.std(per_prompt[cond][c]) / np.sqrt(
                    len(per_prompt[cond][c])
                )
            else:
                stats[cond]["mean"][c] = 0.0
                stats[cond]["std"][c] = 0.0
                stats[cond]["sem"][c] = 0.0

    return stats


def create_visualizations(
    aggregate_counts, operation_counts, all_results, results_dir, num_samples
):
    """Create frequency plots comparing answer distributions with error bars."""
    fontsize_title = 20
    fontsize_label = 18
    fontsize_tick = 16
    fontsize_legend = 12

    categories = ["original_correct", "patched_expected"]
    labels = ["Original Correct", "Patched Expected"]
    x = np.arange(len(categories))
    width = 0.15  # Bar width for 5 conditions

    # =========================================================================
    # Plot 1: Aggregate frequency comparison with std dev
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))

    # Compute statistics across prompts
    stats = compute_per_prompt_stats(all_results, num_samples)

    normal_means = [stats["normal"]["mean"][c] for c in categories]
    normal_sems = [stats["normal"]["sem"][c] for c in categories]
    patched_same_means = [stats["patched_same"]["mean"][c] for c in categories]
    patched_same_sems = [stats["patched_same"]["sem"][c] for c in categories]
    patched_cross_means = [stats["patched_cross"]["mean"][c] for c in categories]
    patched_cross_sems = [stats["patched_cross"]["sem"][c] for c in categories]
    patched_latent_and_resid_means = [
        stats["patched_latent_and_resid"]["mean"][c] for c in categories
    ]
    patched_latent_and_resid_sems = [
        stats["patched_latent_and_resid"]["sem"][c] for c in categories
    ]
    patched_random_means = [stats["patched_random"]["mean"][c] for c in categories]
    patched_random_sems = [stats["patched_random"]["sem"][c] for c in categories]

    ax.bar(
        x - 2 * width,
        normal_means,
        width,
        yerr=normal_sems,
        label="Standard",
        color="#4CAF50",
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
        capsize=3,
    )
    ax.bar(
        x - width,
        patched_same_means,
        width,
        yerr=patched_same_sems,
        label="Patched (same tmpl)",
        color="#e74c3c",
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
        capsize=3,
    )
    ax.bar(
        x,
        patched_cross_means,
        width,
        yerr=patched_cross_sems,
        label="Patched (cross tmpl)",
        color="#3498db",
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
        capsize=3,
    )
    ax.bar(
        x + width,
        patched_latent_and_resid_means,
        width,
        yerr=patched_latent_and_resid_sems,
        label="Patched (latent+resid)",
        color="#9b59b6",
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
        capsize=3,
    )
    ax.bar(
        x + 2 * width,
        patched_random_means,
        width,
        yerr=patched_random_sems,
        label="Patched (random)",
        color="#95a5a6",
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
        capsize=3,
    )

    ax.set_xlabel("Answer Category", fontsize=fontsize_label, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=fontsize_label, fontweight="bold")
    ax.set_title(
        "Changing answer by patching latent vector",
        fontsize=fontsize_title,
        fontweight="bold",
        pad=16,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fontsize_tick)
    ax.tick_params(axis="y", labelsize=fontsize_tick)
    ax.legend(loc="best", fontsize=fontsize_legend)
    ax.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.8)
    ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    output_path = results_dir / "frequency_plot.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close(fig)

    # =========================================================================
    # Plot 1 (counts): Aggregate count comparison with std dev
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))

    # Compute per-prompt count statistics (frequency * num_samples)
    normal_count_means = [stats["normal"]["mean"][c] * num_samples for c in categories]
    normal_count_sems = [stats["normal"]["sem"][c] * num_samples for c in categories]
    patched_same_count_means = [
        stats["patched_same"]["mean"][c] * num_samples for c in categories
    ]
    patched_same_count_sems = [
        stats["patched_same"]["sem"][c] * num_samples for c in categories
    ]
    patched_cross_count_means = [
        stats["patched_cross"]["mean"][c] * num_samples for c in categories
    ]
    patched_cross_count_sems = [
        stats["patched_cross"]["sem"][c] * num_samples for c in categories
    ]
    patched_latent_and_resid_count_means = [
        stats["patched_latent_and_resid"]["mean"][c] * num_samples for c in categories
    ]
    patched_latent_and_resid_count_sems = [
        stats["patched_latent_and_resid"]["sem"][c] * num_samples for c in categories
    ]
    patched_random_count_means = [
        stats["patched_random"]["mean"][c] * num_samples for c in categories
    ]
    patched_random_count_sems = [
        stats["patched_random"]["sem"][c] * num_samples for c in categories
    ]

    ax.bar(
        x - 2 * width,
        normal_count_means,
        width,
        yerr=normal_count_sems,
        label="Standard",
        color="#4CAF50",
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
        capsize=3,
    )
    ax.bar(
        x - width,
        patched_same_count_means,
        width,
        yerr=patched_same_count_sems,
        label="Patched (same tmpl)",
        color="#e74c3c",
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
        capsize=3,
    )
    ax.bar(
        x,
        patched_cross_count_means,
        width,
        yerr=patched_cross_count_sems,
        label="Patched (cross tmpl)",
        color="#3498db",
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
        capsize=3,
    )
    ax.bar(
        x + width,
        patched_latent_and_resid_count_means,
        width,
        yerr=patched_latent_and_resid_count_sems,
        label="Patched (latent+resid)",
        color="#9b59b6",
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
        capsize=3,
    )
    ax.bar(
        x + 2 * width,
        patched_random_count_means,
        width,
        yerr=patched_random_count_sems,
        label="Patched (random)",
        color="#95a5a6",
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
        capsize=3,
    )

    ax.set_xlabel("Answer Category", fontsize=fontsize_label, fontweight="bold")
    ax.set_ylabel("Mean Count per Prompt", fontsize=fontsize_label, fontweight="bold")
    ax.set_title(
        "Changing answer by patching latent vector (Counts)",
        fontsize=fontsize_title,
        fontweight="bold",
        pad=16,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fontsize_tick)
    ax.tick_params(axis="y", labelsize=fontsize_tick)
    ax.legend(loc="best", fontsize=fontsize_legend)
    ax.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.8)

    plt.tight_layout()
    output_path = results_dir / "count_plot.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close(fig)

    # =========================================================================
    # Plot 1b: Per-operation frequency comparison
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax_idx, op in enumerate(["addition", "subtraction"]):
        ax = axes[ax_idx]

        # Compute per-operation statistics from per-prompt data
        op_results = [r for r in all_results if r["operation"] == op]
        if op_results:
            op_stats = compute_per_prompt_stats(op_results, num_samples)

            normal_means = [op_stats["normal"]["mean"][c] for c in categories]
            normal_sems = [op_stats["normal"]["sem"][c] for c in categories]
            patched_same_means = [
                op_stats["patched_same"]["mean"][c] for c in categories
            ]
            patched_same_sems = [op_stats["patched_same"]["sem"][c] for c in categories]
            patched_cross_means = [
                op_stats["patched_cross"]["mean"][c] for c in categories
            ]
            patched_cross_sems = [
                op_stats["patched_cross"]["sem"][c] for c in categories
            ]
            patched_latent_and_resid_means = [
                op_stats["patched_latent_and_resid"]["mean"][c] for c in categories
            ]
            patched_latent_and_resid_sems = [
                op_stats["patched_latent_and_resid"]["sem"][c] for c in categories
            ]
            patched_random_means = [
                op_stats["patched_random"]["mean"][c] for c in categories
            ]
            patched_random_sems = [
                op_stats["patched_random"]["sem"][c] for c in categories
            ]

            ax.bar(
                x - 2 * width,
                normal_means,
                width,
                yerr=normal_sems,
                label="Normal",
                color="#4CAF50",
                alpha=0.8,
                edgecolor="black",
                linewidth=2,
                capsize=3,
            )
            ax.bar(
                x - width,
                patched_same_means,
                width,
                yerr=patched_same_sems,
                label="Same tmpl",
                color="#e74c3c",
                alpha=0.8,
                edgecolor="black",
                linewidth=2,
                capsize=3,
            )
            ax.bar(
                x,
                patched_cross_means,
                width,
                yerr=patched_cross_sems,
                label="Cross tmpl",
                color="#3498db",
                alpha=0.8,
                edgecolor="black",
                linewidth=2,
                capsize=3,
            )
            ax.bar(
                x + width,
                patched_latent_and_resid_means,
                width,
                yerr=patched_latent_and_resid_sems,
                label="Latent+resid",
                color="#9b59b6",
                alpha=0.8,
                edgecolor="black",
                linewidth=2,
                capsize=3,
            )
            ax.bar(
                x + 2 * width,
                patched_random_means,
                width,
                yerr=patched_random_sems,
                label="Random",
                color="#95a5a6",
                alpha=0.8,
                edgecolor="black",
                linewidth=2,
                capsize=3,
            )

        ax.set_xlabel("Answer Category", fontsize=fontsize_label)
        ax.set_ylabel("Frequency", fontsize=fontsize_label)
        ax.set_title(f"{op.capitalize()} Templates", fontsize=fontsize_title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=fontsize_tick)
        ax.tick_params(axis="y", labelsize=fontsize_tick)
        ax.legend(loc="best", fontsize=fontsize_legend - 2)
        ax.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.8)
        ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    output_path = results_dir / "frequency_by_operation.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close(fig)

    # =========================================================================
    # Plot 1b (counts): Per-operation count comparison with std dev
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax_idx, op in enumerate(["addition", "subtraction"]):
        ax = axes[ax_idx]

        # Compute per-operation statistics from per-prompt data
        op_results = [r for r in all_results if r["operation"] == op]
        if op_results:
            op_stats = compute_per_prompt_stats(op_results, num_samples)

            normal_count_means = [
                op_stats["normal"]["mean"][c] * num_samples for c in categories
            ]
            normal_count_sems = [
                op_stats["normal"]["sem"][c] * num_samples for c in categories
            ]
            patched_same_count_means = [
                op_stats["patched_same"]["mean"][c] * num_samples for c in categories
            ]
            patched_same_count_sems = [
                op_stats["patched_same"]["sem"][c] * num_samples for c in categories
            ]
            patched_cross_count_means = [
                op_stats["patched_cross"]["mean"][c] * num_samples for c in categories
            ]
            patched_cross_count_sems = [
                op_stats["patched_cross"]["sem"][c] * num_samples for c in categories
            ]
            patched_latent_and_resid_count_means = [
                op_stats["patched_latent_and_resid"]["mean"][c] * num_samples
                for c in categories
            ]
            patched_latent_and_resid_count_sems = [
                op_stats["patched_latent_and_resid"]["sem"][c] * num_samples
                for c in categories
            ]
            patched_random_count_means = [
                op_stats["patched_random"]["mean"][c] * num_samples for c in categories
            ]
            patched_random_count_sems = [
                op_stats["patched_random"]["sem"][c] * num_samples for c in categories
            ]

            ax.bar(
                x - 2 * width,
                normal_count_means,
                width,
                yerr=normal_count_sems,
                label="Normal",
                color="#4CAF50",
                alpha=0.8,
                edgecolor="black",
                linewidth=2,
                capsize=3,
            )
            ax.bar(
                x - width,
                patched_same_count_means,
                width,
                yerr=patched_same_count_sems,
                label="Same tmpl",
                color="#e74c3c",
                alpha=0.8,
                edgecolor="black",
                linewidth=2,
                capsize=3,
            )
            ax.bar(
                x,
                patched_cross_count_means,
                width,
                yerr=patched_cross_count_sems,
                label="Cross tmpl",
                color="#3498db",
                alpha=0.8,
                edgecolor="black",
                linewidth=2,
                capsize=3,
            )
            ax.bar(
                x + width,
                patched_latent_and_resid_count_means,
                width,
                yerr=patched_latent_and_resid_count_sems,
                label="Latent+resid",
                color="#9b59b6",
                alpha=0.8,
                edgecolor="black",
                linewidth=2,
                capsize=3,
            )
            ax.bar(
                x + 2 * width,
                patched_random_count_means,
                width,
                yerr=patched_random_count_sems,
                label="Random",
                color="#95a5a6",
                alpha=0.8,
                edgecolor="black",
                linewidth=2,
                capsize=3,
            )

        ax.set_xlabel("Answer Category", fontsize=fontsize_label)
        ax.set_ylabel("Mean Count per Prompt", fontsize=fontsize_label)
        ax.set_title(f"{op.capitalize()} Templates (Counts)", fontsize=fontsize_title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=fontsize_tick)
        ax.tick_params(axis="y", labelsize=fontsize_tick)
        ax.legend(loc="best", fontsize=fontsize_legend - 2)
        ax.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.8)

    plt.tight_layout()
    output_path = results_dir / "count_by_operation.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close(fig)

    # =========================================================================
    # Plot 2: Per-template breakdown with std dev
    # =========================================================================
    if all_results:
        per_template_dir = results_dir / "per_template_plots"
        per_template_dir.mkdir(exist_ok=True)

        for result in all_results:
            op = result["operation"]
            t_idx = result["template_idx"]

            fig, ax = plt.subplots(figsize=(14, 6))

            # Compute per-template statistics
            template_stats = compute_template_stats(result["test_results"])

            normal_means = [template_stats["normal"]["mean"][c] for c in categories]
            normal_sems = [template_stats["normal"]["sem"][c] for c in categories]
            patched_same_means = [
                template_stats["patched_same"]["mean"][c] for c in categories
            ]
            patched_same_sems = [
                template_stats["patched_same"]["sem"][c] for c in categories
            ]
            patched_cross_means = [
                template_stats["patched_cross"]["mean"][c] for c in categories
            ]
            patched_cross_sems = [
                template_stats["patched_cross"]["sem"][c] for c in categories
            ]
            patched_latent_and_resid_means = [
                template_stats["patched_latent_and_resid"]["mean"][c]
                for c in categories
            ]
            patched_latent_and_resid_sems = [
                template_stats["patched_latent_and_resid"]["sem"][c] for c in categories
            ]
            patched_random_means = [
                template_stats["patched_random"]["mean"][c] for c in categories
            ]
            patched_random_sems = [
                template_stats["patched_random"]["sem"][c] for c in categories
            ]

            ax.bar(
                x - 2 * width,
                normal_means,
                width,
                yerr=normal_sems,
                label="Normal",
                color="#4CAF50",
                alpha=0.8,
                edgecolor="black",
                linewidth=2,
                capsize=3,
            )
            ax.bar(
                x - width,
                patched_same_means,
                width,
                yerr=patched_same_sems,
                label="Same tmpl",
                color="#e74c3c",
                alpha=0.8,
                edgecolor="black",
                linewidth=2,
                capsize=3,
            )
            ax.bar(
                x,
                patched_cross_means,
                width,
                yerr=patched_cross_sems,
                label="Cross tmpl",
                color="#3498db",
                alpha=0.8,
                edgecolor="black",
                linewidth=2,
                capsize=3,
            )
            ax.bar(
                x + width,
                patched_latent_and_resid_means,
                width,
                yerr=patched_latent_and_resid_sems,
                label="Latent+resid",
                color="#9b59b6",
                alpha=0.8,
                edgecolor="black",
                linewidth=2,
                capsize=3,
            )
            ax.bar(
                x + 2 * width,
                patched_random_means,
                width,
                yerr=patched_random_sems,
                label="Random",
                color="#95a5a6",
                alpha=0.8,
                edgecolor="black",
                linewidth=2,
                capsize=3,
            )

            ax.set_xlabel("Answer Category", fontsize=fontsize_label, fontweight="bold")
            ax.set_ylabel("Frequency", fontsize=fontsize_label, fontweight="bold")
            ax.set_title(
                f"{op.capitalize()} Template {t_idx}: Answer Distribution",
                fontsize=fontsize_title,
                fontweight="bold",
                pad=16,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=fontsize_tick)
            ax.tick_params(axis="y", labelsize=fontsize_tick)
            ax.legend(loc="best", fontsize=fontsize_legend - 2)
            ax.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.8)
            ax.set_ylim(0.0, 1.0)

            plt.tight_layout()
            output_path = per_template_dir / f"{op}_template_{t_idx}.png"
            fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)

        print(f"Saved per-template plots to: {per_template_dir}")

        # Per-template count plots with std dev
        per_template_count_dir = results_dir / "per_template_count_plots"
        per_template_count_dir.mkdir(exist_ok=True)

        for result in all_results:
            op = result["operation"]
            t_idx = result["template_idx"]

            fig, ax = plt.subplots(figsize=(14, 6))

            # Compute per-template statistics
            template_stats = compute_template_stats(result["test_results"])

            normal_count_means = [
                template_stats["normal"]["mean"][c] * num_samples for c in categories
            ]
            normal_count_sems = [
                template_stats["normal"]["sem"][c] * num_samples for c in categories
            ]
            patched_same_count_means = [
                template_stats["patched_same"]["mean"][c] * num_samples
                for c in categories
            ]
            patched_same_count_sems = [
                template_stats["patched_same"]["sem"][c] * num_samples
                for c in categories
            ]
            patched_cross_count_means = [
                template_stats["patched_cross"]["mean"][c] * num_samples
                for c in categories
            ]
            patched_cross_count_sems = [
                template_stats["patched_cross"]["sem"][c] * num_samples
                for c in categories
            ]
            patched_latent_and_resid_count_means = [
                template_stats["patched_latent_and_resid"]["mean"][c] * num_samples
                for c in categories
            ]
            patched_latent_and_resid_count_sems = [
                template_stats["patched_latent_and_resid"]["sem"][c] * num_samples
                for c in categories
            ]
            patched_random_count_means = [
                template_stats["patched_random"]["mean"][c] * num_samples
                for c in categories
            ]
            patched_random_count_sems = [
                template_stats["patched_random"]["sem"][c] * num_samples
                for c in categories
            ]

            ax.bar(
                x - 2 * width,
                normal_count_means,
                width,
                yerr=normal_count_sems,
                label="Normal",
                color="#4CAF50",
                alpha=0.8,
                edgecolor="black",
                linewidth=2,
                capsize=3,
            )
            ax.bar(
                x - width,
                patched_same_count_means,
                width,
                yerr=patched_same_count_sems,
                label="Same tmpl",
                color="#e74c3c",
                alpha=0.8,
                edgecolor="black",
                linewidth=2,
                capsize=3,
            )
            ax.bar(
                x,
                patched_cross_count_means,
                width,
                yerr=patched_cross_count_sems,
                label="Cross tmpl",
                color="#3498db",
                alpha=0.8,
                edgecolor="black",
                linewidth=2,
                capsize=3,
            )
            ax.bar(
                x + width,
                patched_latent_and_resid_count_means,
                width,
                yerr=patched_latent_and_resid_count_sems,
                label="Latent+resid",
                color="#9b59b6",
                alpha=0.8,
                edgecolor="black",
                linewidth=2,
                capsize=3,
            )
            ax.bar(
                x + 2 * width,
                patched_random_count_means,
                width,
                yerr=patched_random_count_sems,
                label="Random",
                color="#95a5a6",
                alpha=0.8,
                edgecolor="black",
                linewidth=2,
                capsize=3,
            )

            ax.set_xlabel("Answer Category", fontsize=fontsize_label, fontweight="bold")
            ax.set_ylabel(
                "Mean Count per Prompt", fontsize=fontsize_label, fontweight="bold"
            )
            ax.set_title(
                f"{op.capitalize()} Template {t_idx}: Answer Distribution (Counts)",
                fontsize=fontsize_title,
                fontweight="bold",
                pad=16,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=fontsize_tick)
            ax.tick_params(axis="y", labelsize=fontsize_tick)
            ax.legend(loc="best", fontsize=fontsize_legend - 2)
            ax.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.8)

            plt.tight_layout()
            output_path = per_template_count_dir / f"{op}_template_{t_idx}.png"
            fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)

        print(f"Saved per-template count plots to: {per_template_count_dir}")

    # =========================================================================
    # Plot 3: Accuracy per step2 value
    # =========================================================================
    if all_results:
        # Collect data grouped by step2_gather value
        step2_conditions = [
            "normal",
            "patched_same",
            "patched_cross",
            "patched_latent_and_resid",
            "patched_random",
        ]
        step2_data = defaultdict(
            lambda: {
                cond: {"original_correct": [], "patched_expected": []}
                for cond in step2_conditions
            }
        )

        for result in all_results:
            for test_result in result["test_results"]:
                step2_gather = test_result["step2_gather"]
                samples = test_result["samples"]
                if not samples:
                    continue

                # Count per-prompt frequencies
                counts = {
                    cond: {"original_correct": 0, "patched_expected": 0}
                    for cond in step2_conditions
                }
                cross_count = 0

                for sample in samples:
                    if sample["category_normal"] == "original_correct":
                        counts["normal"]["original_correct"] += 1
                    elif sample["category_normal"] == "patched_expected":
                        counts["normal"]["patched_expected"] += 1

                    if sample["category_patched_same"] == "original_correct":
                        counts["patched_same"]["original_correct"] += 1
                    elif sample["category_patched_same"] == "patched_expected":
                        counts["patched_same"]["patched_expected"] += 1

                    if (
                        sample["category_patched_latent_and_resid"]
                        == "original_correct"
                    ):
                        counts["patched_latent_and_resid"]["original_correct"] += 1
                    elif (
                        sample["category_patched_latent_and_resid"]
                        == "patched_expected"
                    ):
                        counts["patched_latent_and_resid"]["patched_expected"] += 1

                    if sample["category_patched_random"] == "original_correct":
                        counts["patched_random"]["original_correct"] += 1
                    elif sample["category_patched_random"] == "patched_expected":
                        counts["patched_random"]["patched_expected"] += 1

                    if sample["category_patched_cross"] is not None:
                        cross_count += 1
                        cat = sample["category_patched_cross"]
                        if cat == "original_correct":
                            counts["patched_cross"]["original_correct"] += 1
                        elif cat == "patched_expected":
                            counts["patched_cross"]["patched_expected"] += 1

                n_samples = len(samples)
                for cat in ["original_correct", "patched_expected"]:
                    step2_data[step2_gather]["normal"][cat].append(
                        counts["normal"][cat] / n_samples
                    )
                    step2_data[step2_gather]["patched_same"][cat].append(
                        counts["patched_same"][cat] / n_samples
                    )
                    step2_data[step2_gather]["patched_latent_and_resid"][cat].append(
                        counts["patched_latent_and_resid"][cat] / n_samples
                    )
                    step2_data[step2_gather]["patched_random"][cat].append(
                        counts["patched_random"][cat] / n_samples
                    )
                    if cross_count > 0:
                        step2_data[step2_gather]["patched_cross"][cat].append(
                            counts["patched_cross"][cat] / cross_count
                        )

        # Sort step2 values
        sorted_step2 = sorted(step2_data.keys())

        if len(sorted_step2) > 1:
            # Create plot for "Original Correct" frequency per step2
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))

            for ax_idx, category in enumerate(["original_correct", "patched_expected"]):
                ax = axes[ax_idx]
                cat_label = (
                    "Original Correct"
                    if category == "original_correct"
                    else "Patched Expected"
                )

                # Compute means and SEMs for each step2 value
                cond_stats = {
                    cond: {"means": [], "sems": []} for cond in step2_conditions
                }

                for s2 in sorted_step2:
                    data = step2_data[s2]
                    for cond in step2_conditions:
                        if data[cond][category]:
                            cond_stats[cond]["means"].append(
                                np.mean(data[cond][category])
                            )
                            cond_stats[cond]["sems"].append(
                                np.std(data[cond][category])
                                / np.sqrt(len(data[cond][category]))
                            )
                        else:
                            cond_stats[cond]["means"].append(0)
                            cond_stats[cond]["sems"].append(0)

                x = np.arange(len(sorted_step2))
                width = 0.15

                ax.bar(
                    x - 2 * width,
                    cond_stats["normal"]["means"],
                    width,
                    yerr=cond_stats["normal"]["sems"],
                    label="Normal",
                    color="#4CAF50",
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1.5,
                    capsize=2,
                )
                ax.bar(
                    x - width,
                    cond_stats["patched_same"]["means"],
                    width,
                    yerr=cond_stats["patched_same"]["sems"],
                    label="Same tmpl",
                    color="#e74c3c",
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1.5,
                    capsize=2,
                )
                ax.bar(
                    x,
                    cond_stats["patched_cross"]["means"],
                    width,
                    yerr=cond_stats["patched_cross"]["sems"],
                    label="Cross tmpl",
                    color="#3498db",
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1.5,
                    capsize=2,
                )
                ax.bar(
                    x + width,
                    cond_stats["patched_latent_and_resid"]["means"],
                    width,
                    yerr=cond_stats["patched_latent_and_resid"]["sems"],
                    label="Latent+Resid",
                    color="#9b59b6",
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1.5,
                    capsize=2,
                )
                ax.bar(
                    x + 2 * width,
                    cond_stats["patched_random"]["means"],
                    width,
                    yerr=cond_stats["patched_random"]["sems"],
                    label="Random",
                    color="#95a5a6",
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1.5,
                    capsize=2,
                )

                ax.set_xlabel(
                    "Step2 Value (patched)", fontsize=fontsize_label, fontweight="bold"
                )
                ax.set_ylabel("Frequency", fontsize=fontsize_label, fontweight="bold")
                ax.set_title(
                    f"{cat_label} Rate by Step2 Value",
                    fontsize=fontsize_title,
                    fontweight="bold",
                    pad=10,
                )
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [str(s) for s in sorted_step2], fontsize=fontsize_tick - 2
                )
                ax.tick_params(axis="y", labelsize=fontsize_tick)
                ax.legend(loc="best", fontsize=fontsize_legend - 3)
                ax.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.8)
                ax.set_ylim(0.0, 1.0)

            plt.tight_layout()
            output_path = results_dir / "accuracy_by_step2.png"
            fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"Saved: {output_path}")
            plt.close(fig)

    # =========================================================================
    # Plot 4: Correlation between prompt difficulty and patching success
    # =========================================================================
    if all_results:
        # Collect per-template statistics
        template_stats_list = []
        for result in all_results:
            op = result["operation"]
            t_idx = result["template_idx"]
            template_stats = compute_template_stats(result["test_results"])

            template_stats_list.append(
                {
                    "operation": op,
                    "template_idx": t_idx,
                    "label": f"{op[:3]}_{t_idx}",
                    "original_correct": template_stats["normal"]["mean"].get(
                        "original_correct", 0
                    ),
                    "patched_same": template_stats["patched_same"]["mean"].get(
                        "patched_expected", 0
                    ),
                    "patched_cross": template_stats["patched_cross"]["mean"].get(
                        "patched_expected", 0
                    ),
                    "patched_latent_resid": template_stats["patched_latent_and_resid"][
                        "mean"
                    ].get("patched_expected", 0),
                    "patched_random": template_stats["patched_random"]["mean"].get(
                        "patched_expected", 0
                    ),
                }
            )

        if template_stats_list:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Extract data
            x_orig = [t["original_correct"] for t in template_stats_list]
            y_same = [t["patched_same"] for t in template_stats_list]
            y_cross = [t["patched_cross"] for t in template_stats_list]
            y_latent_resid = [t["patched_latent_resid"] for t in template_stats_list]
            y_random = [t["patched_random"] for t in template_stats_list]
            labels = [t["label"] for t in template_stats_list]
            operations = [t["operation"] for t in template_stats_list]

            # Different markers for addition vs subtraction
            markers = {"addition": "o", "subtraction": "s"}

            # Plot each condition
            for i, (x, ys, yr, yc, ylr, label, op) in enumerate(
                zip(
                    x_orig,
                    y_same,
                    y_random,
                    y_cross,
                    y_latent_resid,
                    labels,
                    operations,
                )
            ):
                marker = markers[op]
                ax.scatter(
                    x, ys, c="#e74c3c", marker=marker, s=100, alpha=0.8, zorder=3
                )
                ax.scatter(
                    x, yc, c="#3498db", marker=marker, s=100, alpha=0.8, zorder=3
                )
                ax.scatter(
                    x, ylr, c="#9b59b6", marker=marker, s=100, alpha=0.8, zorder=3
                )
                ax.scatter(
                    x, yr, c="#95a5a6", marker=marker, s=100, alpha=0.8, zorder=3
                )

            # Add regression lines for each condition
            from scipy import stats as scipy_stats

            conditions_data = [
                (x_orig, y_same, "#e74c3c", "Same tmpl"),
                (x_orig, y_cross, "#3498db", "Cross tmpl"),
                (x_orig, y_latent_resid, "#9b59b6", "Latent+resid"),
                (x_orig, y_random, "#95a5a6", "Random"),
            ]

            for x_data, y_data, color, cond_label in conditions_data:
                # Filter out zeros for regression (in case some conditions have no data)
                valid_mask = [y > 0 or x > 0 for x, y in zip(x_data, y_data)]
                x_valid = [x for x, m in zip(x_data, valid_mask) if m]
                y_valid = [y for y, m in zip(y_data, valid_mask) if m]

                if len(x_valid) >= 2:
                    slope, intercept, r_value, p_value, std_err = (
                        scipy_stats.linregress(x_valid, y_valid)
                    )
                    x_line = np.linspace(min(x_valid), max(x_valid), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(
                        x_line,
                        y_line,
                        color=color,
                        linestyle="--",
                        linewidth=2,
                        alpha=0.7,
                        label=f"{cond_label} (r={r_value:.2f}, p={p_value:.3f})",
                    )

            # Create legend handles for markers
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="gray",
                    markersize=10,
                    label="Addition",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor="gray",
                    markersize=10,
                    label="Subtraction",
                ),
            ]

            ax.set_xlabel(
                "Original Accuracy (Normal Inference)",
                fontsize=fontsize_label,
                fontweight="bold",
            )
            ax.set_ylabel(
                "Patching Success (Patched Expected Rate)",
                fontsize=fontsize_label,
                fontweight="bold",
            )
            ax.set_title(
                "Prompt Difficulty vs Patching Success",
                fontsize=fontsize_title,
                fontweight="bold",
                pad=16,
            )

            # Combine legends
            handles1, labels1 = ax.get_legend_handles_labels()
            ax.legend(
                handles=handles1 + legend_elements,
                loc="best",
                fontsize=fontsize_legend - 2,
            )

            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.8)

            # Add diagonal reference line (y=x would mean perfect correlation)
            ax.plot([0, 1], [0, 1], "k:", alpha=0.3, label="_nolegend_")

            plt.tight_layout()
            output_path = results_dir / "difficulty_vs_patching.png"
            fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"Saved: {output_path}")
            plt.close(fig)

            # Print correlation statistics
            print("\nCorrelation between original accuracy and patching success:")
            for x_data, y_data, color, cond_label in conditions_data:
                valid_mask = [y > 0 or x > 0 for x, y in zip(x_data, y_data)]
                x_valid = [x for x, m in zip(x_data, valid_mask) if m]
                y_valid = [y for y, m in zip(y_data, valid_mask) if m]
                if len(x_valid) >= 2:
                    r, p = scipy_stats.pearsonr(x_valid, y_valid)
                    print(f"  {cond_label}: r={r:.3f}, p={p:.4f}")


# %%
if __name__ == "__main__":
    fire.Fire(main)
