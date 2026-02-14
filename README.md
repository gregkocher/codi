# CODI interpretability

* Our AI Alignment blog post
* Original paper [CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation](https://arxiv.org/abs/2502.21074)


## Setup

```
uv venv
uv pip install -r requirements.txt
```

Set required environment variables in a `.env` file.

## Train a CODI model

```bash
python train.py configs/llama1b_gsm8k-aug-nl.yaml
```

This repository supports also a distributed training with torchrun.
```bash
torchrun --nproc_per_node=4 train.py configs/llama1b_gsm8k-aug-nl.yaml
```

## Evaluate a CODI model

```bash
bash scripts/test_llama1b.sh
```

## Reproduce the experiments in the post

First prepare the prompts as JSON.
```bash
python prepare_prompts_json.py
```

Then execute script experiments. Experiments are sorted according to the order in the blog post.

```bash
python experiments/1_latent_iterations_accuracy.py
python experiments/2_latent_token_probability_numbers.py
python experiments/3_logit_lens_latents.py
python experiments/4_mean_latent_patching_same_vs_diff.py --limit=10
python experiments/5_mean_ablation_combined_templates.py
python experiments/6_operation_probe_latent_vectors.py
```

## Further Experiments

### Running

```bash
# python experiments/10_noisy_latent_rollouts.py
# python experiments/11_latent_interpolation.py
python experiments/12_input_space_interpolation.py --preset=simple_add --x_start=2 --x_end=10 --extra_x='[15,20,50,100,500,1000,5000,10000]'
python experiments/12_input_space_interpolation.py --preset=subtraction --x_start=1 --x_end=10 --extra_x='[20,50,100,500,1000,1845,1997,2026,5000,10000,100000]'
python experiments/12_input_space_interpolation.py --preset=addition
```

### Overview

- **Experiment 10 — Noisy Latent Rollouts**: Runs N rollouts of the same prompt with norm-preserving Gaussian noise injected at specified latent reasoning positions. Compares answers and applies logit lens across rollouts to see how noise changes decoded tokens at each latent step.

- **Experiment 11 — Latent Interpolation**: Interpolates (lerp or slerp) between two prompts' latent reasoning vectors at a chosen position, then completes reasoning from each interpolated starting point. Tests whether intermediate latent vectors produce intermediate answers.

- **Experiment 12 — Input-Space Interpolation**: Varies a single number in the input prompt across a range of values, runs each through the model, and collects all latent vectors and answers. Analyzes with logit lens (per-layer and final-layer heatmaps), t-SNE, UMAP, cosine similarity, vector norms, and cross-layer token stability/entropy plots. Supports three prompt presets: `simple_add`, `addition`, and `subtraction`. See the [Experiment 12 report](reports/experiment_12.md) for a summary of findings.
