# ABOUTME: Plot results from JSON file showing only enabled baselines
# ABOUTME: Filters conditions based on ENABLED_CONDITIONS dictionary

# %%
import json
import numpy as np
import matplotlib.pyplot as plt

# %%
# Configuration
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

JSON_FILE = "/workspace/projects/codi/nov_29_multi_template_results.json"

# %%
# Load results
with open(JSON_FILE, "r") as f:
    results = json.load(f)

# %%
# Map condition names to JSON keys
CONDITION_MAPPING = {
    "baseline": ("baseline", "Baseline", "#2ecc71"),
    "mean_ablated_frozen": ("frozen", "Prompt Mean-Abl\n+ Frozen", "#3498db"),
    "mean_ablated_regenerated": ("regenerated", "Prompt Mean-Abl", "#e74c3c"),
    "mean_ablated_patched": ("patched", "Prompt Mean-Abl\n+ Latent Patched", "#9b59b6"),
    "mean_ablated_cross_template_mean": (
        "cross_template_mean",
        "Prompt Mean-Abl\n+ Latent Patched Cross",
        "#95a5a6",
    ),
    "template_plain": ("template_plain", "Prompt Template", "#f39c12"),
    "template_patched": (
        "template_patched",
        "Prompt Template\n+ Latent Patched",
        "#16a085",
    ),
    "template_frozen": ("template_frozen", "Prompt Template\n+ Frozen", "#e67e22"),
    "template_cross_template_mean": (
        "template_cross_template_mean",
        "Prompt Template\n+ Latent Patched Cross",
        "#34495e",
    ),
}

# %%
# Filter enabled conditions
enabled_data = []
for condition_key, enabled in ENABLED_CONDITIONS.items():
    if enabled:
        json_key, display_name, color = CONDITION_MAPPING[condition_key]
        enabled_data.append(
            {
                "json_key": json_key,
                "display_name": display_name,
                "color": color,
            }
        )

# %%
# Extract overall accuracies for enabled conditions
methods = []
mean_accs = []
std_accs = []
colors = []

for data in enabled_data:
    json_key = data["json_key"]
    overall_mean_key = f"overall_{json_key}_mean_accuracy"
    overall_std_key = f"overall_{json_key}_std_accuracy"

    if overall_mean_key in results and overall_std_key in results:
        methods.append(data["display_name"])
        mean_accs.append(results[overall_mean_key] * 100)
        std_accs.append(results[overall_std_key] * 100)
        colors.append(data["color"])

# %%
# Plot 1: All conditions averaged over all templates
fig1, ax1 = plt.subplots(1, 1, figsize=(16, 6))

bars = ax1.bar(methods, mean_accs, yerr=std_accs, capsize=5, color=colors, alpha=0.8)

ax1.set_ylabel("Accuracy (%)", fontsize=12)
ax1.set_title(
    f"Patching Accuracy",
    fontsize=14,
    fontweight="bold",
)
ax1.set_ylim([0, max(mean_accs) * 1.2 if mean_accs else 100])
ax1.grid(axis="y", alpha=0.3, linestyle="--")

for i, (bar, mean_acc, std_acc) in enumerate(zip(bars, mean_accs, std_accs)):
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + std_acc + 1,
        f"{mean_acc:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

chart_filename1 = "/workspace/projects/codi/nov_29_enabled_baselines_averaged.png"
fig1.savefig(chart_filename1, dpi=300, bbox_inches="tight")
print(f"Chart 1 saved to {chart_filename1}")

# %%
# Plot 2: Per-template results for enabled conditions
fig2, ax2 = plt.subplots(1, 1, figsize=(20, 6))

num_templates = results["num_templates"]
template_labels = [f"T{i + 1}" for i in range(num_templates)]
x = np.arange(len(template_labels))
width = 0.10  # Width of bars

condition_data = []
for data in enabled_data:
    json_key = data["json_key"]
    mean_key = f"{json_key}_mean_accuracy"

    if mean_key in results["per_template_results"][0]:
        template_accs = [tr[mean_key] * 100 for tr in results["per_template_results"]]
        condition_data.append((template_accs, data["display_name"], data["color"]))

width = 0.8 / len(condition_data) if condition_data else 0.10

for i, (data, label, color) in enumerate(condition_data):
    offset = (i - len(condition_data) / 2) * width + width / 2
    bars = ax2.bar(
        x + offset,
        data,
        width,
        label=label,
        color=color,
        alpha=0.8,
    )

ax2.set_xlabel("Template", fontsize=12)
ax2.set_ylabel("Accuracy (%)", fontsize=12)
ax2.set_title(
    f"Per-Template Results for All Conditions\n(Showing {len(condition_data)} enabled baselines)",
    fontsize=14,
    fontweight="bold",
)
ax2.set_xticks(x)
ax2.set_xticklabels(template_labels)
ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))
ax2.grid(axis="y", alpha=0.3, linestyle="--")
ax2.set_ylim([0, 100])

plt.tight_layout()
chart_filename2 = "/workspace/projects/codi/nov_29_enabled_baselines_per_template.png"
fig2.savefig(chart_filename2, dpi=300, bbox_inches="tight")
print(f"Chart 2 saved to {chart_filename2}")

# %%
print("\nEnabled baselines plotted:")
for data in enabled_data:
    json_key = data["json_key"]
    overall_mean_key = f"overall_{json_key}_mean_accuracy"
    if overall_mean_key in results:
        mean_acc = results[overall_mean_key] * 100
        std_acc = results[f"overall_{json_key}_std_accuracy"] * 100
        print(f"  {data['display_name']}: {mean_acc:.2f}% ± {std_acc:.2f}%")
