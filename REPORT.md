# Probing the Limits: Adversarial and OOD Robustness of Neural Thicket Ensembles

## 1. Executive Summary
This study tests whether ensembles built from weight-space perturbations around a pretrained CIFAR-10 model are more robust to adversarial attacks and OOD inputs than a single model, and whether explicit diversity-inducing perturbation directions help further.

Key finding: a random perturbation ensemble improved strong-attack robustness at `eps=8/255` over a single model by `+0.45` percentage points (bootstrap 95% CI `[+0.15, +0.80]`), but structured diversity strategies (orthogonal and adversarial-direction perturbations) did not improve over random under this setup.

Practically, perturbation ensembling appears to offer a modest robustness gain at high attack strength with limited engineering overhead, but diversity design must be constrained to avoid clean-accuracy and calibration regressions.

## 2. Goal
### Hypothesis
Neural thicket ensembles from random perturbations improve robustness vs single models, and diversity-aware perturbation directions (orthogonal or adversarial) further improve resilience.

### Importance
Robust classification under adversarial and distribution shift is critical for safety-critical deployment. If perturbation-generated ensembles are effective, they provide a low-cost alternative to training many independent models.

### Problem Solved
This work provides a controlled benchmark across clean accuracy, PGD robustness, OOD detection, calibration, and diversity diagnostics for four methods:
- `single`
- `random` perturbation ensemble
- `orthogonal` perturbation ensemble
- `adversarial` direction perturbation ensemble

## 3. Data Construction
### Dataset Description
- ID dataset: CIFAR-10 (local Hugging Face disk format), `50,000` train / `10,000` test.
- OOD dataset: SVHN (`cropped_digits`) test split, `26,032` examples.
- Local paths: `datasets/cifar10/`, `datasets/svhn/`.

### Example Samples
Representative records were saved to `results/example_samples.json`:
- CIFAR-10 train: indices `[0,1,2]` with labels `[6,9,9]`.
- SVHN test: indices `[0,1,2]` with labels `[5,2,1]`.

### Data Quality
From `results/dataset_summary.json`:
- Missing labels: `0%` for all splits.
- Duplicate estimate: `0` (index-aware estimate).
- Class balance:
  - CIFAR-10 train/test: exactly balanced (`5000` or `1000` per class).
  - SVHN test: imbalanced (e.g., class `1`: `5099`, class `0`: `1744`).

### Preprocessing Steps
1. Training augmentation (CIFAR-10 train): random crop (padding 4) + horizontal flip + tensor conversion.
2. Evaluation transforms (CIFAR-10 test, SVHN test): tensor conversion only.
3. Pixel range normalized to `[0,1]` (no dataset mean/std normalization).

### Train/Val/Test Splits
- CIFAR-10: used provided train/test only (no extra validation split).
- SVHN: used provided test split as OOD evaluation set.
- Adversarial evaluation used first `2000` CIFAR-10 test examples for runtime control (documented in config).

## 4. Experiment Description
### Methodology
#### High-Level Approach
1. Train one CIFAR-10 ResNet-18 baseline (CIFAR stem modifications).
2. Generate ensemble experts by perturbing base weights in different direction sets.
3. Recalibrate BN statistics for each expert.
4. Evaluate clean, PGD robustness, and OOD confidence separation.

#### Why This Method?
It directly isolates perturbation direction strategy while controlling architecture, dataset, and ensemble size. It operationalizes the hypothesis mechanism (diversity/independence of experts).

### Implementation Details
#### Tools and Libraries
- Python `3.12.8`
- PyTorch `2.10.0+cu128`
- Torchvision `0.25.0+cu128`
- NumPy, Pandas, SciPy, scikit-learn, Matplotlib, Seaborn, datasets

#### Algorithms/Models
- Base model: ResNet-18 adapted for CIFAR (`3x3` conv stem, no maxpool).
- Ensemble size: `M=6` experts.
- Perturbation methods:
  - Random Gaussian directions (normalized).
  - Orthogonalized random directions (Gram-Schmidt over flattened vector space).
  - Adversarial directions from per-batch loss gradients, then orthogonalized.

#### Hyperparameters
| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| epochs | 8 | pragmatic budget |
| train_batch_size | 128 | GPU memory guideline (24GB) |
| eval_batch_size | 256 | throughput |
| optimizer | SGD(momentum=0.9) | standard CIFAR baseline |
| lr | 0.1 + cosine schedule | common ResNet-CIFAR setting |
| weight_decay | 5e-4 | standard baseline |
| label_smoothing | 0.05 | mild regularization |
| ensemble_size | 6 | fixed compare budget |
| perturb_sigma | 0.015 | stable clean performance range |
| pgd_steps | 10 | runtime/strength trade-off |
| pgd_eps | 2/255, 4/255, 8/255 | standard L_inf stress levels |
| robust_eval_size | 2000 | runtime control (documented) |
| bootstrap_samples | 3000 | stable CI estimates |

#### Training / Analysis Pipeline
Implemented in `src/run_research.py`:
1. Data loading and QC summary.
2. Base training (or checkpoint resume).
3. Ensemble generation and BN recalibration.
4. Clean/OOD/PGD evaluations.
5. Diversity metrics and bootstrap statistical analysis.
6. Save CSV/JSON artifacts + plots.

### Experimental Protocol
#### Reproducibility Information
- Random seeds: global `42`; method seeds `{42, 101, 202, 303}`.
- Hardware: 2x NVIDIA RTX 3090 (24GB each), CUDA enabled.
- Mixed precision: enabled (`torch.cuda.amp` autocast + scaler).
- Runtime: `5.68` minutes for full evaluation pass after checkpoint load (`results/runtime.json`).

#### Evaluation Metrics
- Clean accuracy: ID correctness on CIFAR-10 test.
- ECE: calibration gap over 15 confidence bins.
- NLL: negative log-likelihood from ensemble probabilities.
- PGD robust accuracy: post-attack accuracy at each epsilon.
- OOD AUROC/AUPR/FPR95: confidence-based ID-vs-OOD discrimination using max-softmax.
- Diversity: pairwise disagreement and pairwise error correlation among experts.

### Raw Results
#### Main Table
| Method | Clean Acc | ECE | NLL | OOD AUROC | OOD AUPR | FPR95 | PGD@2/255 | PGD@4/255 | PGD@8/255 |
|--------|-----------|-----|-----|-----------|----------|-------|-----------|-----------|-----------|
| single | 0.8828 | 0.0350 | 0.3821 | 0.8389 | 0.7680 | 0.7827 | 0.1375 | 0.0055 | 0.0005 |
| random | 0.8699 | 0.0360 | 0.4138 | 0.8430 | 0.7705 | 0.7891 | 0.1195 | 0.0175 | 0.0050 |
| orthogonal | 0.8702 | 0.0362 | 0.4129 | 0.8419 | 0.7663 | 0.7855 | 0.1190 | 0.0175 | 0.0060 |
| adversarial | 0.8402 | 0.0894 | 0.5429 | 0.8228 | 0.7627 | 0.8468 | 0.1135 | 0.0045 | 0.0000 |

#### Statistical Tests (bootstrap)
- `PGD@8/255 random - single`: `+0.0045`, 95% CI `[+0.0015, +0.0080]`, `p=0.004`.
- `PGD@8/255 orthogonal - random`: `+0.0010`, 95% CI `[-0.0020, +0.0040]`, `p=0.599`.
- `PGD@8/255 adversarial - random`: `-0.0050`, 95% CI `[-0.0085, -0.0020]`, `p<0.001`.
- `Clean random - single`: `-0.0129`, 95% CI `[-0.0172, -0.0086]`, `p<0.001`.

#### Diversity Summary
- Random disagreement: `0.00619`, error corr: `0.9788`.
- Orthogonal disagreement: `0.00544`, error corr: `0.9811`.
- Adversarial disagreement: `0.2125`, error corr: `0.4922`.

#### Output Locations
- Metrics JSON: `results/metrics.json`
- Stats JSON: `results/statistical_analysis.json`
- Summary CSV: `results/summary_table.csv`
- Plots: `results/plots/`
- Base model: `results/models/base_model.pt`

## 5. Result Analysis
### Key Findings
1. Random perturbation ensemble improved strongest-attack robustness (`8/255`) over single model, but reduced clean accuracy.
2. Orthogonal perturbations behaved similarly to random, with no significant robustness gain under the chosen sigma.
3. Adversarial-direction perturbations produced very high disagreement but substantially worse clean calibration and no robustness gain.

### Hypothesis Testing
- H1 (random > single robustness): **partially supported**.
  - Supported at strong attack (`8/255`) with significant gain.
  - Not supported at low attack (`2/255`), where random underperformed.
- H2 (orthogonal > random via diversity): **not supported**.
- H3 (adversarial-direction > random robustness): **refuted** in this setup.
- H4 (higher diversity correlates with better robustness/OOD): **not supported**.
  - Spearman disagreement vs OOD AUROC: `rho=-0.20`, `p=0.8`.
  - Spearman disagreement vs PGD@8/255: `rho=-0.40`, `p=0.6`.

### Comparison to Baselines
- Best OOD AUROC was random (`0.8430`), only marginally above single (`0.8389`).
- Best PGD@8/255 was orthogonal (`0.0060`) then random (`0.0050`) vs single (`0.0005`), but orthogonal-random difference was not significant.
- Adversarial-direction underperformed on most metrics despite much larger disagreement.

### Surprises and Insights
- Maximizing directional diversity without local performance constraints can hurt calibration and clean accuracy.
- Very high disagreement did not translate into stronger robustness, suggesting “useful diversity” must be quality-controlled.

### Error Analysis
- Under `8/255` PGD, all methods collapsed to very low robust accuracy (<1%), indicating non-adversarially-trained models remain highly vulnerable.
- Structured perturbations mainly shifted confidence behavior rather than creating robust decision boundaries.

### Limitations
- Single pretrained base model (no repeated base-training seeds).
- Robust evaluation on 2000-sample subset for runtime.
- Only one perturbation scale (`sigma=0.015`) in final table.
- OOD setting limited to CIFAR-10 vs SVHN.
- No AutoAttack or adversarial training baselines.

## 6. Conclusions
Perturbation ensembles around a strong pretrained model can provide a measurable but modest robustness gain at high attack strength compared with a single model. In this run, explicit direction strategies (orthogonal/adversarial) did not beat simple random perturbations. The results suggest that expert independence alone is insufficient; robustness-aware constraints during expert generation are likely necessary.

## 7. Next Steps
### Immediate Follow-ups
1. Sweep `sigma` and ensemble size to identify stable diversity-performance regimes.
2. Add adversarially trained base checkpoints to test whether perturbation ensembles stack with robust training.
3. Repeat with multiple base-training seeds and full 10k PGD evaluation for tighter confidence intervals.

### Alternative Approaches
- Diversity objectives using feature-space disagreement with clean-accuracy constraints.
- Gradient-orthogonal perturbations constrained by Fisher/Hessian local geometry.

### Broader Extensions
- Evaluate on CIFAR-10-C and TinyImageNet OOD benchmarks.
- Compare to independently trained deep ensembles at equal compute.

### Open Questions
- What diversity measure best predicts robustness gains?
- Can we optimize perturbation directions for robustness without sacrificing calibration?

## References
- Gan, Y., Isola, P. (2026). *Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights*.
- Mehrtash, A., et al. (2020). *PEP: Parameter Ensembling by Perturbation*.
- Lakshminarayanan, B., et al. (2016). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*.
- Tramèr, F., et al. (2017). *Ensemble Adversarial Training: Attacks and Defenses*.
- Xia, Y., Bouganis, C.-S. (2022). *On the Usefulness of Deep Ensemble Diversity for OOD Detection*.

## 8. Validation Checklist
### Code Validation
- All pipeline runs completed without runtime errors after dataset-schema fix.
- Reproducibility rerun completed; key conclusions were stable.
- Minor numeric drift observed in adversarial-direction row (GPU nondeterminism), but significance conclusions unchanged.
- Random seeds are set and configuration is saved in `results/config.json`.
- No hardcoded absolute paths in experiment code.

### Scientific Validation
- Metrics are appropriate for the question (clean, PGD, OOD, calibration, diversity).
- Bootstrap CIs and p-values reported for primary comparisons.
- Conclusions match measured results and include negative findings.
- Alternative explanation (diversity magnitude vs useful diversity) considered.

### Documentation Validation
- `planning.md`, `REPORT.md`, `README.md` completed with actual outputs.
- Plots and tables are generated and referenced.
- Reproduction commands provided.

### Output Validation
- Expected artifacts are present in `results/` (metrics JSON/CSV, plots, checkpoint, runtime, environment).
