# ABOUTME: Test if latent vectors from concrete prompts transfer to abstract templates.
# ABOUTME: Runs multiple trials with sampling, plots match rate and accuracy with error bars.
# %%
import csv
import re

import matplotlib.pyplot as plt
import numpy as np
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
GREEDY = False  # Use sampling instead of greedy

# Sampling parameters
NUM_SAMPLES_PER_PROMPT = 10

# Different prompt variations (X, Y, Z values)
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
# RUN EXPERIMENT TRIALS
# =============================================================================
# %%
results_by_prompt = []
all_individual_results = []  # Store all individual samples for CSV export

print("\n" + "=" * 80)
print("RUNNING LATENT TRANSFER EXPERIMENT WITH SAMPLING")
print("=" * 80)

for prompt_idx, (X, Y, Z) in enumerate(PROMPT_VARIATIONS, 1):
    correct_answer = calculate_correct_answer(X, Y, Z)
    print(f"\n{'=' * 80}")
    print(f"PROMPT {prompt_idx}/{len(PROMPT_VARIATIONS)}: X={X}, Y={Y}, Z={Z}")
    print(f"Correct answer: {correct_answer}")
    print("=" * 80)

    matches = []
    concrete_correct = []
    baseline_correct = []

    for sample_idx in range(NUM_SAMPLES_PER_PROMPT):
        print(f"\n  Sample {sample_idx + 1}/{NUM_SAMPLES_PER_PROMPT}")

        # =====================================================================
        # INFERENCE 1: Concrete prompt with actual numbers
        # =====================================================================
        prompt_concrete = f"""A team starts with {X} members. They recruit {Y} new members. Then each current member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else."""

        inputs_concrete = tokenizer(prompt_concrete, return_tensors="pt", padding=True)
        input_ids_concrete = inputs_concrete["input_ids"].to(model.codi.device)
        attention_mask_concrete = inputs_concrete["attention_mask"].to(
            model.codi.device
        )

        output_concrete = model.generate(
            input_ids=input_ids_concrete,
            attention_mask=attention_mask_concrete,
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

        generated_concrete = tokenizer.decode(
            output_concrete["sequences"][0], skip_special_tokens=False
        )
        latent_vectors_concrete = output_concrete["latent_vectors"]

        # Extract answer from concrete output
        concrete_answer = extract_number_from_output(generated_concrete)
        is_concrete_correct = concrete_answer == correct_answer

        print(f"    Concrete: {concrete_answer} (correct: {is_concrete_correct})")

        # =====================================================================
        # INFERENCE 2: Abstract prompt with X/Y/Z, using latent vectors
        # =====================================================================
        prompt_abstract = """A team starts with  X members. They recruit  Y new members. Then each current member recruits  Z additional people. How many people are there now on the team? Give the answer only and nothing else."""

        inputs_abstract = tokenizer(prompt_abstract, return_tensors="pt", padding=True)
        input_ids_abstract = inputs_abstract["input_ids"].to(model.codi.device)
        attention_mask_abstract = inputs_abstract["attention_mask"].to(
            model.codi.device
        )
        assert input_ids_concrete.shape == input_ids_abstract.shape

        output_abstract = model.generate(
            input_ids=input_ids_abstract,
            attention_mask=attention_mask_abstract,
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
            latent_vectors_override=latent_vectors_concrete,
        )

        generated_abstract = tokenizer.decode(
            output_abstract["sequences"][0], skip_special_tokens=False
        )

        # Extract answer from abstract output
        abstract_answer = extract_number_from_output(generated_abstract)
        outputs_match = concrete_answer == abstract_answer

        print(f"    Abstract: {abstract_answer} (match: {outputs_match})")

        # =====================================================================
        # INFERENCE 3: Baseline - Abstract prompt WITHOUT latent vectors
        # =====================================================================
        output_baseline = model.generate(
            input_ids=input_ids_abstract,
            attention_mask=attention_mask_abstract,
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

        # Extract answer from baseline output
        baseline_answer = extract_number_from_output(generated_baseline)
        is_baseline_correct = baseline_answer == correct_answer

        print(f"    Baseline: {baseline_answer} (correct: {is_baseline_correct})")

        matches.append(outputs_match)
        concrete_correct.append(is_concrete_correct)
        baseline_correct.append(is_baseline_correct)

        # Save individual result for CSV export
        all_individual_results.append(
            {
                "prompt_idx": prompt_idx,
                "sample_idx": sample_idx + 1,
                "X": X,
                "Y": Y,
                "Z": Z,
                "correct_answer": correct_answer,
                "concrete_output": generated_concrete,
                "concrete_answer": concrete_answer,
                "concrete_correct": is_concrete_correct,
                "abstract_output": generated_abstract,
                "abstract_answer": abstract_answer,
                "outputs_match": outputs_match,
                "baseline_output": generated_baseline,
                "baseline_answer": baseline_answer,
                "baseline_correct": is_baseline_correct,
            }
        )

    # Calculate statistics for this prompt
    match_rate = np.mean(matches)
    match_std = np.std(matches)
    accuracy_rate = np.mean(concrete_correct)
    accuracy_std = np.std(concrete_correct)
    baseline_accuracy_rate = np.mean(baseline_correct)
    baseline_accuracy_std = np.std(baseline_correct)

    results_by_prompt.append(
        {
            "X": X,
            "Y": Y,
            "Z": Z,
            "correct_answer": correct_answer,
            "match_rate": match_rate,
            "match_std": match_std,
            "accuracy_rate": accuracy_rate,
            "accuracy_std": accuracy_std,
            "baseline_accuracy_rate": baseline_accuracy_rate,
            "baseline_accuracy_std": baseline_accuracy_std,
            "matches": matches,
            "concrete_correct": concrete_correct,
            "baseline_correct": baseline_correct,
        }
    )

    print(
        f"\n  Summary: Match rate = {match_rate:.2%} ± {match_std:.2%}, "
        f"Accuracy = {accuracy_rate:.2%} ± {accuracy_std:.2%}, "
        f"Baseline = {baseline_accuracy_rate:.2%} ± {baseline_accuracy_std:.2%}"
    )

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
# %%
print("\n" + "=" * 80)
print("EXPERIMENT SUMMARY")
print("=" * 80)

overall_match_rate = np.mean([r["match_rate"] for r in results_by_prompt])
overall_accuracy_rate = np.mean([r["accuracy_rate"] for r in results_by_prompt])
overall_baseline_accuracy_rate = np.mean(
    [r["baseline_accuracy_rate"] for r in results_by_prompt]
)

print(f"\nOverall match rate: {overall_match_rate:.2%}")
print(f"Overall accuracy rate (concrete): {overall_accuracy_rate:.2%}")
print(f"Overall baseline accuracy rate: {overall_baseline_accuracy_rate:.2%}")

print("\n" + "-" * 80)
print("DETAILED RESULTS PER PROMPT")
print("-" * 80)

for i, r in enumerate(results_by_prompt, 1):
    print(
        f"\nPrompt {i}: X={r['X']}, Y={r['Y']}, Z={r['Z']} (correct={r['correct_answer']})"
    )
    print(f"  Match rate: {r['match_rate']:.2%} ± {r['match_std']:.2%}")
    print(f"  Accuracy rate: {r['accuracy_rate']:.2%} ± {r['accuracy_std']:.2%}")
    print(
        f"  Baseline accuracy: {r['baseline_accuracy_rate']:.2%} ± {r['baseline_accuracy_std']:.2%}"
    )

# =============================================================================
# CREATE PLOTS
# =============================================================================
# %%
print("\n" + "=" * 80)
print("CREATING PLOTS")
print("=" * 80)

# Prepare data for plotting
prompt_labels = [f"({r['X']},{r['Y']},{r['Z']})" for r in results_by_prompt]
match_rates = [r["match_rate"] * 100 for r in results_by_prompt]
match_stds = [r["match_std"] * 100 for r in results_by_prompt]
accuracy_rates = [r["accuracy_rate"] * 100 for r in results_by_prompt]
accuracy_stds = [r["accuracy_std"] * 100 for r in results_by_prompt]
baseline_accuracy_rates = [r["baseline_accuracy_rate"] * 100 for r in results_by_prompt]
baseline_accuracy_stds = [r["baseline_accuracy_std"] * 100 for r in results_by_prompt]

# Create figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1: Match rate
x_pos = np.arange(len(prompt_labels))
ax1.bar(x_pos, match_rates, alpha=0.7, color="steelblue")
ax1.set_xlabel("Prompt (X, Y, Z)", fontsize=12)
ax1.set_ylabel("Match Rate (%)", fontsize=12)
ax1.set_title(
    "Match Rate: Concrete vs Abstract Outputs\n(with same latent vectors)", fontsize=14
)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(prompt_labels, rotation=45, ha="right")
ax1.set_ylim(0, 100)
ax1.axhline(
    y=overall_match_rate * 100,
    color="red",
    linestyle="--",
    alpha=0.5,
    label=f"Overall: {overall_match_rate:.1%}",
)
ax1.legend()
ax1.grid(axis="y", alpha=0.3)

# Plot 2: Accuracy rate
ax2.bar(x_pos, accuracy_rates, alpha=0.7, color="forestgreen")
ax2.set_xlabel("Prompt (X, Y, Z)", fontsize=12)
ax2.set_ylabel("Accuracy Rate (%)", fontsize=12)
ax2.set_title("Accuracy: Concrete Outputs\n(correct answer rate)", fontsize=14)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(prompt_labels, rotation=45, ha="right")
ax2.set_ylim(0, 100)
ax2.axhline(
    y=overall_accuracy_rate * 100,
    color="red",
    linestyle="--",
    alpha=0.5,
    label=f"Overall: {overall_accuracy_rate:.1%}",
)
ax2.legend()
ax2.grid(axis="y", alpha=0.3)

# Plot 3: Baseline accuracy rate
ax3.bar(x_pos, baseline_accuracy_rates, alpha=0.7, color="coral")
ax3.set_xlabel("Prompt (X, Y, Z)", fontsize=12)
ax3.set_ylabel("Baseline Accuracy Rate (%)", fontsize=12)
ax3.set_title(
    "Baseline: Abstract Prompt WITHOUT Latent Vectors\n(correct answer rate)",
    fontsize=14,
)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(prompt_labels, rotation=45, ha="right")
ax3.set_ylim(0, 100)
ax3.axhline(
    y=overall_baseline_accuracy_rate * 100,
    color="red",
    linestyle="--",
    alpha=0.5,
    label=f"Overall: {overall_baseline_accuracy_rate:.1%}",
)
ax3.legend()
ax3.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("latent_transfer_results.png", dpi=300, bbox_inches="tight")
print("Plot saved to: latent_transfer_results.png")
plt.close()

# Create a combined plot with three bars
fig, ax = plt.subplots(figsize=(16, 6))

width = 0.25
x_pos = np.arange(len(prompt_labels))


bars1 = ax.bar(
    x_pos,
    accuracy_rates,
    width,
    alpha=0.7,
    color="forestgreen",
    label="Standard prompt w/ latent vectors",
)

bars2 = ax.bar(
    x_pos - width,
    match_rates,
    width,
    alpha=0.7,
    color="steelblue",
    label="Template prompt w/ latent vectors",
)

bars3 = ax.bar(
    x_pos + width,
    baseline_accuracy_rates,
    width,
    alpha=0.7,
    color="coral",
    label="Template prompt w/o latent vectors",
)

ax.set_xlabel("Prompt (X, Y, Z)", fontsize=12)
ax.set_ylabel("Rate (%)", fontsize=12)
ax.set_title(
    f"Latent Vector Transfer: Match Rate vs Accuracy vs Baseline\n(sampling with T=1.0, {NUM_SAMPLES_PER_PROMPT} samples per prompt)",
    fontsize=14,
)
ax.set_xticks(x_pos)
ax.set_xticklabels(prompt_labels, rotation=45, ha="right")
ax.set_ylim(0, 110)
ax.legend(fontsize=12)
ax.grid(axis="y", alpha=0.3)

# Add overall averages as horizontal lines
ax.axhline(
    y=overall_match_rate * 100,
    color="steelblue",
    linestyle="--",
    alpha=0.5,
    linewidth=1,
)
ax.axhline(
    y=overall_accuracy_rate * 100,
    color="forestgreen",
    linestyle="--",
    alpha=0.5,
    linewidth=1,
)

plt.tight_layout()
plt.savefig("latent_transfer_combined.png", dpi=300, bbox_inches="tight")
print("Combined plot saved to: latent_transfer_combined.png")
plt.close()

# Create overall averaged results plot
fig, ax = plt.subplots(figsize=(10, 6))

categories = [
    "Standard prompt w/ latent vectors",
    "Template prompt w/ latent vectors",
    "Template prompt w/o latent vectors",
]
values = [
    overall_accuracy_rate * 100,
    overall_match_rate * 100,
    overall_baseline_accuracy_rate * 100,
]
colors = ["steelblue", "forestgreen", "coral"]

bars = ax.bar(categories, values, alpha=0.7, color=colors, width=0.5)

# Add value labels on top of bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )

ax.set_ylabel("Rate (%)", fontsize=12)
ax.set_title(
    f"Overall Results Averaged Across All {len(PROMPT_VARIATIONS)} Prompt Variations\n(T=1.0, {NUM_SAMPLES_PER_PROMPT} samples per prompt)",
    fontsize=14,
)
ax.set_ylim(0, 110)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("latent_transfer_overall.png", dpi=300, bbox_inches="tight")
print("Overall averaged plot saved to: latent_transfer_overall.png")
plt.close()

print("\n" + "=" * 80)
print("FINAL RESULTS:")
print(f"  Overall Match Rate: {overall_match_rate:.2%}")
print(f"  Overall Concrete Accuracy Rate: {overall_accuracy_rate:.2%}")
print(f"  Overall Baseline Accuracy Rate: {overall_baseline_accuracy_rate:.2%}")
print("=" * 80)

# =============================================================================
# SAVE RESULTS TO CSV
# =============================================================================
# %%
print("\n" + "=" * 80)
print("SAVING RESULTS TO CSV")
print("=" * 80)

csv_filename = "latent_transfer_results.csv"
csv_fields = [
    "prompt_idx",
    "sample_idx",
    "X",
    "Y",
    "Z",
    "correct_answer",
    "concrete_answer",
    "concrete_correct",
    "abstract_answer",
    "outputs_match",
    "baseline_answer",
    "baseline_correct",
    "concrete_output",
    "abstract_output",
    "baseline_output",
]

with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
    writer.writeheader()
    writer.writerows(all_individual_results)

print(f"Results saved to: {csv_filename}")
print(f"Total rows: {len(all_individual_results)}")

# %%
