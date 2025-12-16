# ABOUTME: Test if patching latent vectors can change intermediate computations and final answers.
# ABOUTME: Compares addition prompts (x+y=5) with subtraction prompts, patching latent vectors between them.
# %%
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import re
import csv

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
TEMPERATURE = 0.0  # Use greedy decoding for consistency
TOP_K = 40
TOP_P = 0.95
GREEDY = True

# Which latent vector index to patch (0-indexed)
# Index 1 = second latent vector, which should correspond to first intermediate computation
LATENT_INDEX_TO_PATCH = 2

# Phase 1: Addition prompts where x + y = 5
# These will be used to compute the average "5" latent vector
ADDITION_VARIATIONS = [
    (1, 4, 1),  # 1+4=5, then 5*2=10
    (2, 3, 1),  # 2+3=5, then 5*2=10
    (3, 2, 1),  # 3+2=5, then 5*2=10
    (4, 1, 1),  # 4+1=5, then 5*2=10
]

# Phase 2: Subtraction prompts where x - y ≠ 5
# We'll patch these with the "5" vector to see if answers change
SUBTRACTION_VARIATIONS = [
    (8, 2, 1),  # 8-2=6, then 6*2=12 (without patch)
    (7, 1, 1),  # 7-1=6, then 6*2=12 (without patch)
    (9, 3, 1),  # 9-3=6, then 6*2=12 (without patch)
    (10, 4, 1),  # 10-4=6, then 6*2=12 (without patch)
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def calculate_answer_addition(X, Y, Z):
    """Calculate: (X+Y) * (1+Z)"""
    return (X + Y) * (1 + Z)


def calculate_answer_subtraction(X, Y, Z):
    """Calculate: (X-Y) * (1+Z)"""
    return (X - Y) * (1 + Z)


def extract_number_from_output(text):
    """Extract the first number from the model output"""
    numbers = re.findall(r"\d+", text)
    if numbers:
        return int(numbers[0])
    return None


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
# PHASE 1: COLLECT AND AVERAGE LATENT VECTORS FROM ADDITION PROMPTS
# =============================================================================
# %%
print("\n" + "=" * 80)
print("PHASE 1: COLLECTING LATENT VECTORS FROM ADDITION PROMPTS (x+y=5)")
print("=" * 80)

addition_latent_vectors = []

for idx, (X, Y, Z) in enumerate(ADDITION_VARIATIONS, 1):
    correct_answer = calculate_answer_addition(X, Y, Z)
    print(f"\nAddition prompt {idx}/{len(ADDITION_VARIATIONS)}: X={X}, Y={Y}, Z={Z}")
    print(f"  Expected: {X}+{Y}={X + Y}, then {X + Y}*(1+{Z})={correct_answer}")

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

    generated = tokenizer.decode(output["sequences"][0], skip_special_tokens=False)
    answer = extract_number_from_output(generated)
    latent_vectors = output["latent_vectors"]

    print(f"  Model answer: {answer} (correct: {answer == correct_answer})")
    print(f"  Latent vectors shape: {[lv.shape for lv in latent_vectors]}")

    addition_latent_vectors.append(latent_vectors)

# Average the latent vectors across all addition prompts
print("\n" + "-" * 80)
print("Computing average latent vector...")

# Stack all latent vectors for each iteration
# Note: there are NUM_LATENT_ITERATIONS + 1 vectors (initial + one per iteration)
num_vectors = len(addition_latent_vectors[0])
averaged_latent_vectors = []
for iteration_idx in range(num_vectors):
    # Collect all latent vectors for this iteration
    latents_for_iteration = [lv[iteration_idx] for lv in addition_latent_vectors]
    # Stack and average
    stacked = np.stack([lv.cpu().float().numpy() for lv in latents_for_iteration])
    averaged = np.mean(stacked, axis=0)
    averaged_latent_vectors.append(averaged)
    print(f"  Iteration {iteration_idx}: shape {averaged.shape}")

print(f"\nAveraged latent vectors computed for all {num_vectors} vectors")
print(f"Will patch latent vector at index {LATENT_INDEX_TO_PATCH}")

# =============================================================================
# PHASE 2: TEST PATCHING ON SUBTRACTION PROMPTS
# =============================================================================
# %%
print("\n" + "=" * 80)
print("PHASE 2: TESTING LATENT VECTOR PATCHING ON SUBTRACTION PROMPTS")
print("=" * 80)

results = []

for idx, (X, Y, Z) in enumerate(SUBTRACTION_VARIATIONS, 1):
    unpatched_answer = calculate_answer_subtraction(X, Y, Z)
    # If we successfully patch to make intermediate = 5, expected answer is 5*(1+Z)
    patched_expected_answer = 5 * (1 + Z)

    print(f"\n{'-' * 80}")
    print(
        f"Subtraction prompt {idx}/{len(SUBTRACTION_VARIATIONS)}: X={X}, Y={Y}, Z={Z}"
    )
    print(f"  Without patch: {X}-{Y}={X - Y}, then {X - Y}*(1+{Z})={unpatched_answer}")
    print(f"  If patched to 5: 5*(1+{Z})={patched_expected_answer}")

    prompt = f"""A team starts with {X} members. They remove {Y} members. Then each current member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else."""

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(model.codi.device)
    attention_mask = inputs["attention_mask"].to(model.codi.device)

    # -------------------------------------------------------------------------
    # BASELINE: Normal inference without patching
    # -------------------------------------------------------------------------
    print("\n  [BASELINE] Running without patching...")
    output_baseline = model.generate(
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

    generated_baseline = tokenizer.decode(
        output_baseline["sequences"][0], skip_special_tokens=False
    )
    answer_baseline = extract_number_from_output(generated_baseline)
    baseline_latent_vectors = output_baseline["latent_vectors"]
    print(f"  [BASELINE] Answer: {answer_baseline}")

    # -------------------------------------------------------------------------
    # PATCHED: Inference with latent vector patching
    # -------------------------------------------------------------------------
    print(
        f"\n  [PATCHED] Running with latent vector {LATENT_INDEX_TO_PATCH} patched..."
    )

    # Prepare patched latent vectors: copy baseline vectors but replace one
    import torch

    num_baseline_vectors = len(baseline_latent_vectors)
    patched_latent_vectors = []
    for i in range(num_baseline_vectors):
        if i == LATENT_INDEX_TO_PATCH:
            # Replace this vector with the averaged "5" vector
            patched_latent_vectors.append(
                torch.tensor(
                    averaged_latent_vectors[i],
                    device=model.codi.device,
                    dtype=model.codi.dtype,
                )
            )
        else:
            # Keep the baseline vector
            patched_latent_vectors.append(baseline_latent_vectors[i].clone())

    print(
        f"    Replaced latent vector at index {LATENT_INDEX_TO_PATCH}, kept {num_baseline_vectors - 1} others from baseline"
    )

    output_patched = model.generate(
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
        latent_vectors_override=patched_latent_vectors,
    )

    generated_patched = tokenizer.decode(
        output_patched["sequences"][0], skip_special_tokens=False
    )
    answer_patched = extract_number_from_output(generated_patched)
    print(f"  [PATCHED] Answer: {answer_patched}")

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------
    answer_changed = answer_baseline != answer_patched
    matches_expected = answer_patched == patched_expected_answer

    print(f"\n  [ANALYSIS]")
    print(
        f"    Answer changed: {answer_changed} ({answer_baseline} -> {answer_patched})"
    )
    print(
        f"    Matches expected patched answer ({patched_expected_answer}): {matches_expected}"
    )

    results.append(
        {
            "X": X,
            "Y": Y,
            "Z": Z,
            "baseline_answer": answer_baseline,
            "patched_answer": answer_patched,
            "expected_unpatched": unpatched_answer,
            "expected_patched": patched_expected_answer,
            "answer_changed": answer_changed,
            "matches_expected_patched": matches_expected,
            "baseline_output": generated_baseline,
            "patched_output": generated_patched,
        }
    )

# =============================================================================
# SUMMARY
# =============================================================================
# %%
print("\n" + "=" * 80)
print("EXPERIMENT SUMMARY")
print("=" * 80)

total_changed = sum(r["answer_changed"] for r in results)
total_matches_expected = sum(r["matches_expected_patched"] for r in results)

print(f"\nTotal prompts tested: {len(results)}")
print(
    f"Answers changed by patching: {total_changed}/{len(results)} ({100 * total_changed / len(results):.1f}%)"
)
print(
    f"Patched answers match expected: {total_matches_expected}/{len(results)} ({100 * total_matches_expected / len(results):.1f}%)"
)

print("\n" + "-" * 80)
print("DETAILED RESULTS")
print("-" * 80)

for i, r in enumerate(results, 1):
    print(f"\n{i}. X={r['X']}, Y={r['Y']}, Z={r['Z']}")
    print(f"   Baseline: {r['baseline_answer']} (expected: {r['expected_unpatched']})")
    print(f"   Patched:  {r['patched_answer']} (expected: {r['expected_patched']})")
    print(
        f"   Changed: {r['answer_changed']}, Matches expected: {r['matches_expected_patched']}"
    )

# =============================================================================
# SAVE RESULTS TO CSV
# =============================================================================
# %%
print("\n" + "=" * 80)
print("SAVING RESULTS TO CSV")
print("=" * 80)

csv_filename = "latent_vector_patching_results.csv"
csv_fields = [
    "X",
    "Y",
    "Z",
    "baseline_answer",
    "patched_answer",
    "expected_unpatched",
    "expected_patched",
    "answer_changed",
    "matches_expected_patched",
    "baseline_output",
    "patched_output",
]

with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
    writer.writeheader()
    writer.writerows(results)

print(f"Results saved to: {csv_filename}")

# =============================================================================
# CREATE VISUALIZATION
# =============================================================================
# %%
print("\n" + "=" * 80)
print("CREATING VISUALIZATION")
print("=" * 80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Baseline vs Patched answers
prompt_labels = [f"({r['X']},{r['Y']},{r['Z']})" for r in results]
x_pos = np.arange(len(prompt_labels))
width = 0.35

baseline_answers = [r["baseline_answer"] for r in results]
patched_answers = [r["patched_answer"] for r in results]
expected_patched = [r["expected_patched"] for r in results]

ax1.bar(
    x_pos - width / 2,
    baseline_answers,
    width,
    label="Baseline (no patch)",
    alpha=0.7,
    color="coral",
)
ax1.bar(
    x_pos + width / 2,
    patched_answers,
    width,
    label="Patched",
    alpha=0.7,
    color="steelblue",
)
ax1.plot(
    x_pos,
    expected_patched,
    "g--",
    marker="o",
    label="Expected (if patch works)",
    linewidth=2,
)

ax1.set_xlabel("Prompt (X, Y, Z)", fontsize=12)
ax1.set_ylabel("Model Answer", fontsize=12)
ax1.set_title("Effect of Latent Vector Patching on Model Answers", fontsize=14)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(prompt_labels, rotation=45, ha="right")
ax1.legend()
ax1.grid(axis="y", alpha=0.3)

# Plot 2: Success metrics
metrics = ["Answers Changed", "Matches Expected"]
values = [
    100 * total_changed / len(results),
    100 * total_matches_expected / len(results),
]
colors = ["steelblue", "forestgreen"]

ax2.bar(metrics, values, alpha=0.7, color=colors, width=0.5)
ax2.set_ylabel("Percentage (%)", fontsize=12)
ax2.set_title("Patching Success Metrics", fontsize=14)
ax2.set_ylim(0, 110)
ax2.grid(axis="y", alpha=0.3)

# Add value labels on bars
for i, (metric, value) in enumerate(zip(metrics, values)):
    ax2.text(i, value + 2, f"{value:.1f}%", ha="center", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("latent_vector_patching_results.png", dpi=300, bbox_inches="tight")
print("Plot saved to: latent_vector_patching_results.png")
plt.close()

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE!")
print("=" * 80)

# %%
