# ABOUTME: Test if averaged activations from multiple prompt variations can be used for inference.
# ABOUTME: Averages layer activations across prompt variations using hooks and compares accuracy.
# %%
import csv
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
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
TEMPERATURE = 1.0
TOP_K = 40
TOP_P = 0.95
GREEDY = True  # Use sampling with temperature

# Sampling parameters
NUM_SAMPLES_PER_PROMPT = 1

# Prompt variations (X, Y, Z values) - will be used for both averaging and testing
PROMPT_VARIATIONS = [
    (4, 2, 4),
    (3, 4, 4),
    (5, 1, 3),
    (2, 3, 2),
    (6, 2, 5),
    (3, 3, 3),
    (4, 4, 2),
    (5, 5, 1),
    (2, 2, 2),
    (2, 4, 2),
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def calculate_correct_answer(X, Y, Z):
    """Calculate the correct answer: (X+Y) + ((X+Y)*Z) = (X+Y) * (1+Z)"""
    return (X + Y) * (1 + Z)


def extract_number_from_output(text):
    """Extract the first number from the model output"""
    numbers = re.findall(r"\d+", text)
    if numbers:
        return int(numbers[0])
    return None


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

    inputs = tokenizer(prompt, return_tensors="pt", padding=False, add_special_tokens=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Add BOCOT token
    bot_id = tokenizer.convert_tokens_to_ids("<|bocot|>")
    input_ids_bot = torch.cat([input_ids, torch.tensor([[bot_id]], device=device)], dim=1)
    attention_mask_bot = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=1)

    return input_ids_bot, attention_mask_bot


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
            output_hidden_states=True
        )

    for h in handles:
        h.remove()

    # Stack activations: (NumLayers, PromptSeqLen, HiddenDim)
    seq_len = input_ids.shape[1] - 1  # Exclude BOCOT token
    prefill_tensor = torch.stack([x[0, :seq_len, :] for x in captured_prefill])

    return {
        "prefill": prefill_tensor,
        "input_ids": input_ids[:, :seq_len]
    }


def compute_mean_prompt_activations(model, tokenizer, prompt_variations):
    """Compute mean activations across multiple prompt variations."""
    print(f"\nComputing mean prompt activations across {len(prompt_variations)} prompts...")

    all_activations = []
    all_seq_lens = []

    for X, Y, Z in prompt_variations:
        prompt = f"""A team starts with {X} members. They recruit {Y} new members. Then each current member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else."""
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
        "max_seq_len": max_seq_len
    }


def _create_mean_activation_hook(layer_idx, mean_prefill, prompt_seq_len, max_seq_len, device):
    """Create a hook function that replaces prompt activations with mean activations."""
    def hook(module, args, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            new_hidden = hidden_states.clone()
            seq_len = min(hidden_states.shape[1], prompt_seq_len, max_seq_len)
            for token_idx in range(seq_len):
                new_hidden[:, token_idx, :] = mean_prefill[layer_idx, token_idx, :].unsqueeze(0).to(device)
            return (new_hidden,) + output[1:]
        else:
            new_hidden = output.clone()
            seq_len = min(output.shape[1], prompt_seq_len, max_seq_len)
            for token_idx in range(seq_len):
                new_hidden[:, token_idx, :] = mean_prefill[layer_idx, token_idx, :].unsqueeze(0).to(device)
            return new_hidden
    return hook


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
# PHASE 1: COMPUTE MEAN ACTIVATIONS AND COLLECT LATENT VECTORS FROM MULTIPLE PROMPTS
# =============================================================================
# %%
print("\n" + "=" * 80)
print("PHASE 1: COMPUTING MEAN ACTIVATIONS AND COLLECTING LATENT VECTORS")
print("=" * 80)

# Compute mean activations across all prompt variations
mean_activations_dict = compute_mean_prompt_activations(
    model, tokenizer, PROMPT_VARIATIONS
)
print(f"\nComputed mean activations across {len(PROMPT_VARIATIONS)} prompts")
print(f"Mean activation shape: {mean_activations_dict['mean_prefill'].shape}")
print(f"Max sequence length: {mean_activations_dict['max_seq_len']}")

# Collect latent vectors for each specific prompt
collected_latent_vectors = {}  # Dict mapping (X, Y, Z) to latent vectors

print("\nCollecting latent vectors for each prompt...")
for prompt_idx, (X, Y, Z) in enumerate(PROMPT_VARIATIONS, 1):
    correct_answer = calculate_correct_answer(X, Y, Z)
    print(
        f"Processing prompt {prompt_idx}/{len(PROMPT_VARIATIONS)}: X={X}, Y={Y}, Z={Z}"
    )

    prompt = f"""A team starts with {X} members. They recruit {Y} new members. Then each current member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else."""

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(model.codi.device)
    attention_mask = inputs["attention_mask"].to(model.codi.device)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
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

    # Store latent vectors
    collected_latent_vectors[(X, Y, Z)] = output["latent_vectors"]

    # Check accuracy
    generated = tokenizer.decode(output["sequences"][0], skip_special_tokens=False)
    answer = extract_number_from_output(generated)
    is_correct = answer == correct_answer
    print(f"  Answer: {answer} (correct: {is_correct})")

print(f"\nCollected latent vectors for {len(collected_latent_vectors)} prompts")

# =============================================================================
# PHASE 2: TEST WITH MEAN ACTIVATIONS ON SAME PROMPTS
# =============================================================================
# %%
print("\n" + "=" * 80)
print("PHASE 2: TESTING WITH MEAN ACTIVATIONS ON SAME PROMPTS")
print("=" * 80)

# Store results by position across all prompts
position_results = {
    "standard": [[] for _ in range(NUM_SAMPLES_PER_PROMPT)],
    "mean_activations": [[] for _ in range(NUM_SAMPLES_PER_PROMPT)],
    "mean_activations_with_latents": [[] for _ in range(NUM_SAMPLES_PER_PROMPT)],
}

results = []
all_individual_results = []

for test_idx, (X, Y, Z) in enumerate(PROMPT_VARIATIONS, 1):
    correct_answer = calculate_correct_answer(X, Y, Z)
    print(f"\n{'=' * 80}")
    print(f"TEST PROMPT {test_idx}/{len(PROMPT_VARIATIONS)}: X={X}, Y={Y}, Z={Z}")
    print(f"Correct answer: {correct_answer}")
    print("=" * 80)

    standard_correct = []
    mean_activations_correct = []
    mean_activations_with_latents_correct = []

    # Get latent vectors for this specific prompt
    prompt_latent_vectors = collected_latent_vectors[(X, Y, Z)]

    for sample_idx in range(NUM_SAMPLES_PER_PROMPT):
        print(f"\n  Sample {sample_idx + 1}/{NUM_SAMPLES_PER_PROMPT}")

        # =====================================================================
        # INFERENCE 1: Standard inference (without averaged past_key_values)
        # =====================================================================
        prompt = f"""A team starts with {X} members. They recruit {Y} new members. Then each current member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else."""

        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(model.codi.device)
        attention_mask = inputs["attention_mask"].to(model.codi.device)

        output_standard = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            num_latent_iterations=NUM_LATENT_ITERATIONS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            greedy=GREEDY,
            return_latent_vectors=False,
            remove_eos=False,
            output_attentions=False,
            skip_thinking=False,
            verbalize_cot=False,
            output_hidden_states=True,
            sot_token=tokenizer.convert_tokens_to_ids("<|bocot|>"),
            eot_token=tokenizer.convert_tokens_to_ids("<|eocot|>"),
        )
        # prompt_latent_vectors = output_standard["latent_vectors"]

        generated_standard = tokenizer.decode(
            output_standard["sequences"][0], skip_special_tokens=False
        )
        standard_answer = extract_number_from_output(generated_standard)
        is_standard_correct = standard_answer == correct_answer

        print(f"    Standard: {standard_answer} (correct: {is_standard_correct})")

        # =====================================================================
        # INFERENCE 2: Using mean activations
        # =====================================================================
        # Set up hooks to replace prompt activations with mean activations
        layers = get_transformer_layers(model)
        mean_prefill = mean_activations_dict["mean_prefill"].to(model.codi.device)
        max_seq_len = mean_activations_dict["max_seq_len"]
        prompt_seq_len = input_ids.shape[1] - 1  # Exclude BOCOT

        patch_handles = []
        for layer_idx, layer in enumerate(layers):
            hook = _create_mean_activation_hook(
                layer_idx, mean_prefill, prompt_seq_len, max_seq_len, model.codi.device
            )
            patch_handles.append(layer.register_forward_hook(hook))

        output_mean_activations = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            num_latent_iterations=NUM_LATENT_ITERATIONS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            greedy=GREEDY,
            return_latent_vectors=False,
            remove_eos=False,
            output_attentions=False,
            skip_thinking=False,
            verbalize_cot=False,
            output_hidden_states=True,
            sot_token=tokenizer.convert_tokens_to_ids("<|bocot|>"),
            eot_token=tokenizer.convert_tokens_to_ids("<|eocot|>"),
        )

        # Remove hooks
        for h in patch_handles:
            h.remove()

        generated_mean_activations = tokenizer.decode(
            output_mean_activations["sequences"][0], skip_special_tokens=False
        )
        mean_activations_answer = extract_number_from_output(generated_mean_activations)
        is_mean_activations_correct = mean_activations_answer == correct_answer

        print(
            f"    Mean Activations: {mean_activations_answer} (correct: {is_mean_activations_correct})"
        )

        # =====================================================================
        # INFERENCE 3: Using mean activations + prompt-specific latent vectors
        # =====================================================================
        # Set up hooks again
        patch_handles = []
        for layer_idx, layer in enumerate(layers):
            hook = _create_mean_activation_hook(
                layer_idx, mean_prefill, prompt_seq_len, max_seq_len, model.codi.device
            )
            patch_handles.append(layer.register_forward_hook(hook))

        output_mean_activations_with_latents = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            num_latent_iterations=NUM_LATENT_ITERATIONS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            greedy=GREEDY,
            return_latent_vectors=False,
            remove_eos=False,
            output_attentions=False,
            skip_thinking=False,
            verbalize_cot=False,
            output_hidden_states=True,
            sot_token=tokenizer.convert_tokens_to_ids("<|bocot|>"),
            eot_token=tokenizer.convert_tokens_to_ids("<|eocot|>"),
            latent_vectors_override=prompt_latent_vectors,  # Use prompt-specific latents
        )

        # Remove hooks
        for h in patch_handles:
            h.remove()

        generated_mean_activations_with_latents = tokenizer.decode(
            output_mean_activations_with_latents["sequences"][0], skip_special_tokens=False
        )
        mean_activations_with_latents_answer = extract_number_from_output(
            generated_mean_activations_with_latents
        )
        is_mean_activations_with_latents_correct = (
            mean_activations_with_latents_answer == correct_answer
        )

        print(
            f"    Mean Activations + Latents: {mean_activations_with_latents_answer} (correct: {is_mean_activations_with_latents_correct})"
        )

        standard_correct.append(is_standard_correct)
        mean_activations_correct.append(is_mean_activations_correct)
        mean_activations_with_latents_correct.append(is_mean_activations_with_latents_correct)

        # Store results by position (for position-wise statistics)
        position_results["standard"][sample_idx].append(is_standard_correct)
        position_results["mean_activations"][sample_idx].append(is_mean_activations_correct)
        position_results["mean_activations_with_latents"][sample_idx].append(
            is_mean_activations_with_latents_correct
        )

        # Save individual result
        all_individual_results.append(
            {
                "test_idx": test_idx,
                "sample_idx": sample_idx + 1,
                "X": X,
                "Y": Y,
                "Z": Z,
                "correct_answer": correct_answer,
                "standard_output": generated_standard,
                "standard_answer": standard_answer,
                "standard_correct": is_standard_correct,
                "mean_activations_output": generated_mean_activations,
                "mean_activations_answer": mean_activations_answer,
                "mean_activations_correct": is_mean_activations_correct,
                "mean_activations_with_latents_output": generated_mean_activations_with_latents,
                "mean_activations_with_latents_answer": mean_activations_with_latents_answer,
                "mean_activations_with_latents_correct": is_mean_activations_with_latents_correct,
            }
        )

    # Calculate statistics for this test prompt
    standard_accuracy = np.mean(standard_correct)
    standard_std = np.std(standard_correct)
    mean_activations_accuracy = np.mean(mean_activations_correct)
    mean_activations_std = np.std(mean_activations_correct)
    mean_activations_with_latents_accuracy = np.mean(mean_activations_with_latents_correct)
    mean_activations_with_latents_std = np.std(mean_activations_with_latents_correct)

    results.append(
        {
            "X": X,
            "Y": Y,
            "Z": Z,
            "correct_answer": correct_answer,
            "standard_accuracy": standard_accuracy,
            "standard_std": standard_std,
            "mean_activations_accuracy": mean_activations_accuracy,
            "mean_activations_std": mean_activations_std,
            "mean_activations_with_latents_accuracy": mean_activations_with_latents_accuracy,
            "mean_activations_with_latents_std": mean_activations_with_latents_std,
            "standard_correct": standard_correct,
            "mean_activations_correct": mean_activations_correct,
            "mean_activations_with_latents_correct": mean_activations_with_latents_correct,
        }
    )

    print(
        f"\n  Summary: Standard = {standard_accuracy:.2%} ± {standard_std:.2%}, "
        f"Mean Activations = {mean_activations_accuracy:.2%} ± {mean_activations_std:.2%}, "
        f"Mean Activations + Latents = {mean_activations_with_latents_accuracy:.2%} ± {mean_activations_with_latents_std:.2%}"
    )

# =============================================================================
# SUMMARY STATISTICS (POSITION-WISE)
# =============================================================================
# %%
print("\n" + "=" * 80)
print("EXPERIMENT SUMMARY (POSITION-WISE STATISTICS)")
print("=" * 80)

# Calculate position-wise accuracies (average across all prompts for each position)
position_wise_accuracies = {
    "standard": [],
    "mean_activations": [],
    "mean_activations_with_latents": [],
}

for position_idx in range(NUM_SAMPLES_PER_PROMPT):
    # For each position, calculate mean accuracy across all prompts
    position_wise_accuracies["standard"].append(
        np.mean(position_results["standard"][position_idx])
    )
    position_wise_accuracies["mean_activations"].append(
        np.mean(position_results["mean_activations"][position_idx])
    )
    position_wise_accuracies["mean_activations_with_latents"].append(
        np.mean(position_results["mean_activations_with_latents"][position_idx])
    )

# Calculate overall mean and std from position-wise accuracies
overall_standard_accuracy = np.mean(position_wise_accuracies["standard"])
overall_standard_std = np.std(position_wise_accuracies["standard"])

overall_mean_activations_accuracy = np.mean(position_wise_accuracies["mean_activations"])
overall_mean_activations_std = np.std(position_wise_accuracies["mean_activations"])

overall_mean_activations_with_latents_accuracy = np.mean(
    position_wise_accuracies["mean_activations_with_latents"]
)
overall_mean_activations_with_latents_std = np.std(
    position_wise_accuracies["mean_activations_with_latents"]
)

print(
    f"\nOverall standard accuracy: {overall_standard_accuracy:.2%} ± {overall_standard_std:.2%}"
)
print(
    f"Overall mean activations accuracy: {overall_mean_activations_accuracy:.2%} ± {overall_mean_activations_std:.2%}"
)
print(
    f"Overall mean activations + Latents accuracy: {overall_mean_activations_with_latents_accuracy:.2%} ± {overall_mean_activations_with_latents_std:.2%}"
)
print(
    f"Difference (Mean Activations vs Standard): {(overall_mean_activations_accuracy - overall_standard_accuracy):.2%}"
)
print(
    f"Difference (Mean Activations + Latents vs Standard): {(overall_mean_activations_with_latents_accuracy - overall_standard_accuracy):.2%}"
)

print("\n" + "-" * 80)
print("POSITION-WISE ACCURACIES")
print("-" * 80)
print(f"{'Position':<10} {'Standard':<15} {'Mean Act':<15} {'Mean Act + Latents':<20}")
print("-" * 80)
for i in range(NUM_SAMPLES_PER_PROMPT):
    print(
        f"{i + 1:<10} {position_wise_accuracies['standard'][i]:.2%} {' ' * 7} "
        f"{position_wise_accuracies['mean_activations'][i]:.2%} {' ' * 7} "
        f"{position_wise_accuracies['mean_activations_with_latents'][i]:.2%}"
    )

print("\n" + "-" * 80)
print("DETAILED RESULTS PER TEST PROMPT")
print("-" * 80)

for i, r in enumerate(results, 1):
    print(
        f"\nTest {i}: X={r['X']}, Y={r['Y']}, Z={r['Z']} (correct={r['correct_answer']})"
    )
    print(
        f"  Standard accuracy: {r['standard_accuracy']:.2%} ± {r['standard_std']:.2%}"
    )
    print(
        f"  Mean Activations accuracy: {r['mean_activations_accuracy']:.2%} ± {r['mean_activations_std']:.2%}"
    )
    print(
        f"  Mean Activations + Latents accuracy: {r['mean_activations_with_latents_accuracy']:.2%} ± {r['mean_activations_with_latents_std']:.2%}"
    )

# =============================================================================
# CREATE PLOTS
# =============================================================================
# %%
print("\n" + "=" * 80)
print("CREATING PLOTS")
print("=" * 80)

# Prepare data for plotting
test_labels = [f"({r['X']},{r['Y']},{r['Z']})" for r in results]
standard_accuracies = [r["standard_accuracy"] * 100 for r in results]
standard_stds = [r["standard_std"] * 100 for r in results]
mean_activations_accuracies = [r["mean_activations_accuracy"] * 100 for r in results]
mean_activations_stds = [r["mean_activations_std"] * 100 for r in results]
mean_activations_with_latents_accuracies = [
    r["mean_activations_with_latents_accuracy"] * 100 for r in results
]
mean_activations_with_latents_stds = [
    r["mean_activations_with_latents_std"] * 100 for r in results
]

# Plot 1: Comparison bar chart
fig, ax = plt.subplots(figsize=(16, 6))

width = 0.25
x_pos = np.arange(len(test_labels))

bars1 = ax.bar(
    x_pos - width,
    standard_accuracies,
    width,
    alpha=0.7,
    color="steelblue",
    label="Standard",
)

bars2 = ax.bar(
    x_pos,
    mean_activations_accuracies,
    width,
    alpha=0.7,
    color="coral",
    label="Mean Activations",
)

bars3 = ax.bar(
    x_pos + width,
    mean_activations_with_latents_accuracies,
    width,
    alpha=0.7,
    color="forestgreen",
    label="Mean Activations + Prompt Latents",
)

ax.set_xlabel("Test Prompt (X, Y, Z)", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title(
    f"Accuracy Comparison: Three Strategies\n(T={TEMPERATURE}, {NUM_SAMPLES_PER_PROMPT} samples per prompt)",
    fontsize=14,
)
ax.set_xticks(x_pos)
ax.set_xticklabels(test_labels, rotation=45, ha="right")
ax.set_ylim(0, 110)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

# Add overall averages as horizontal lines
ax.axhline(
    y=overall_standard_accuracy * 100,
    color="steelblue",
    linestyle="--",
    alpha=0.5,
    linewidth=1,
)
ax.axhline(
    y=overall_mean_activations_accuracy * 100,
    color="coral",
    linestyle="--",
    alpha=0.5,
    linewidth=1,
)
ax.axhline(
    y=overall_mean_activations_with_latents_accuracy * 100,
    color="forestgreen",
    linestyle="--",
    alpha=0.5,
    linewidth=1,
)

plt.tight_layout()
plt.savefig("mean_activations_comparison.png", dpi=300, bbox_inches="tight")
print("Plot saved to: mean_activations_comparison.png")
plt.close()

# Plot 2: Overall averaged results with error bars
fig, ax = plt.subplots(figsize=(12, 6))

categories = [
    "Standard",
    "Mean Activations",
    "Mean Activations +\nPrompt Latents",
]
values = [
    overall_standard_accuracy * 100,
    overall_mean_activations_accuracy * 100,
    overall_mean_activations_with_latents_accuracy * 100,
]
errors = [
    overall_standard_std * 100,
    overall_mean_activations_std * 100,
    overall_mean_activations_with_latents_std * 100,
]
colors = ["steelblue", "coral", "forestgreen"]

bars = ax.bar(
    categories, values, alpha=0.7, color=colors, width=0.5, yerr=errors, capsize=10
)

# Add value labels on top of bars
for bar, value, error in zip(bars, values, errors):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + error + 1,
        f"{value:.1f}% ± {error:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title(
    f"Overall Accuracy: Three Strategies (Position-wise Statistics)\n"
    f"Mean activations from {len(PROMPT_VARIATIONS)} prompts, tested on same {len(PROMPT_VARIATIONS)} prompts\n"
    f"Error bars show std dev across {NUM_SAMPLES_PER_PROMPT} sampling positions (T={TEMPERATURE})",
    fontsize=13,
)
ax.set_ylim(0, 110)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("mean_activations_overall.png", dpi=300, bbox_inches="tight")
print("Overall averaged plot saved to: mean_activations_overall.png")
plt.show()
plt.close()

# Plot 3: Position-wise accuracy trends
fig, ax = plt.subplots(figsize=(14, 6))

positions = np.arange(1, NUM_SAMPLES_PER_PROMPT + 1)

ax.plot(
    positions,
    [acc * 100 for acc in position_wise_accuracies["standard"]],
    marker="o",
    linewidth=2,
    markersize=8,
    color="steelblue",
    label="Standard",
)
ax.plot(
    positions,
    [acc * 100 for acc in position_wise_accuracies["mean_activations"]],
    marker="s",
    linewidth=2,
    markersize=8,
    color="coral",
    label="Mean Activations",
)
ax.plot(
    positions,
    [acc * 100 for acc in position_wise_accuracies["mean_activations_with_latents"]],
    marker="^",
    linewidth=2,
    markersize=8,
    color="forestgreen",
    label="Mean Activations + Prompt Latents",
)

ax.set_xlabel("Sampling Position", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title(
    f"Accuracy Across Sampling Positions\n"
    f"Each point shows mean accuracy across {len(PROMPT_VARIATIONS)} prompts for that position",
    fontsize=14,
)
ax.set_xticks(positions)
ax.set_ylim(0, 110)
ax.legend(fontsize=11, loc="best")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("mean_activations_position_trends.png", dpi=300, bbox_inches="tight")
print("Position trends plot saved to: mean_activations_position_trends.png")
plt.show()
plt.close()

print("\n" + "=" * 80)
print("FINAL RESULTS (POSITION-WISE STATISTICS):")
print(
    f"  Overall Standard Accuracy: {overall_standard_accuracy:.2%} ± {overall_standard_std:.2%}"
)
print(
    f"  Overall Mean Activations Accuracy: {overall_mean_activations_accuracy:.2%} ± {overall_mean_activations_std:.2%}"
)
print(
    f"  Overall Mean Activations + Latents Accuracy: {overall_mean_activations_with_latents_accuracy:.2%} ± {overall_mean_activations_with_latents_std:.2%}"
)
print(
    f"  Difference (Mean Activations vs Standard): {(overall_mean_activations_accuracy - overall_standard_accuracy):.2%}"
)
print(
    f"  Difference (Mean Activations + Latents vs Standard): {(overall_mean_activations_with_latents_accuracy - overall_standard_accuracy):.2%}"
)
print(
    f"\n  Note: Error bars represent std dev across {NUM_SAMPLES_PER_PROMPT} sampling positions"
)
print("=" * 80)

# =============================================================================
# SAVE RESULTS TO CSV
# =============================================================================
# %%
print("\n" + "=" * 80)
print("SAVING RESULTS TO CSV")
print("=" * 80)

csv_filename = "mean_activations_results.csv"
csv_fields = [
    "test_idx",
    "sample_idx",
    "X",
    "Y",
    "Z",
    "correct_answer",
    "standard_answer",
    "standard_correct",
    "mean_activations_answer",
    "mean_activations_correct",
    "mean_activations_with_latents_answer",
    "mean_activations_with_latents_correct",
    "standard_output",
    "mean_activations_output",
    "mean_activations_with_latents_output",
]

with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
    writer.writeheader()
    writer.writerows(all_individual_results)

print(f"Results saved to: {csv_filename}")
print(f"Total rows: {len(all_individual_results)}")

# %%
