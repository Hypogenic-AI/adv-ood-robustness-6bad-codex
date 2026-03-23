# Resources Catalog

## Summary
This document catalogs papers, datasets, and code repositories gathered for experiments on adversarial and OOD robustness of neural thicket ensembles.

## Papers
Total papers downloaded: 14 PDF files (12 core unique papers + 2 duplicate-named copies from initial retrieval pass).

| Title | Authors | Year | File | Key Info |
|------|---------|------|------|---------|
| Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights | Gan, Isola | 2026 | `papers/2603.12228_Neural_Thickets_Diverse_Task_Experts_Are_Dense_Around_Pretrained_Weights_2026.pdf` | Core thicket method |
| Robust Deep Learning Ensemble against Deception | Wei, Liu | 2020 | `papers/2009.06589_Robust_Deep_Learning_Ensemble_against_Deception_2020.pdf` | Joint adversarial+OOD defense |
| Ensemble Adversarial Training: Attacks and Defenses | Tramèr et al. | 2017 | `papers/1705.07204_Ensemble_Adversarial_Training_Attacks_and_Defenses_2017.pdf` | Canonical adversarial ensemble baseline |
| Deep Ensembles for Uncertainty | Lakshminarayanan et al. | 2016 | `papers/1612.01474_Simple_and_Scalable_Predictive_Uncertainty_Deep_Ensembles_2016.pdf` | Standard ensemble baseline |
| OOD Detection Using Leave-out Ensemble | Vyas et al. | 2018 | `papers/1809.03576_OOD_Detection_Ensemble_Self_Supervised_Leave_out_Classifiers_2018.pdf` | OOD-specific ensemble method |
| Enhancing Certifiable Robustness via Deep Model Ensemble | Zhang et al. | 2019 | `papers/1910.14655_Enhancing_Certifiable_Robustness_via_Deep_Model_Ensemble_2019.pdf` | Certified robustness |
| Certifying Joint Adversarial Robustness for Model Ensembles | Ahmad, Evans | 2020 | `papers/2004.10250_Certifying_Joint_Adversarial_Robustness_for_Model_Ensembles_2020.pdf` | Joint certification |
| PEP: Parameter Ensembling by Perturbation | Mehrtash et al. | 2020 | `papers/2010.12721_PEP_Parameter_Ensembling_by_Perturbation_2020.pdf` | Perturbation ensemble precursor |
| Sequential Bayesian Neural Subnetwork Ensembles | Jantre et al. | 2022 | `papers/2206.00794_Sequential_Bayesian_Neural_Subnetwork_Ensembles_2022.pdf` | Efficient Bayesian ensembles |
| Batch-Ensemble Stochastic NNs for OOD | Chen et al. | 2022 | `papers/2206.12911_Batch_Ensemble_Stochastic_NNs_for_OOD_2022.pdf` | OOD with shared parameters |
| Usefulness of Deep Ensemble Diversity for OOD | Xia, Bouganis | 2022 | `papers/2207.07517_Usefulness_of_Deep_Ensemble_Diversity_for_OOD_2022.pdf` | Diversity-vs-OOD analysis |
| Hierarchical Pruning with Focal Diversity | Wu et al. | 2023 | `papers/2311.10293_Hierarchical_Pruning_of_Deep_Ensembles_with_Focal_Diversity_2023.pdf` | Diversity-aware pruning |

See `papers/README.md` for detailed descriptions.

## Datasets
Total datasets downloaded: 2

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| CIFAR-10 | Hugging Face `cifar10` | 50k train / 10k test (~131 MB local) | Image classification | `datasets/cifar10/` | Primary ID dataset |
| SVHN (`cropped_digits`) | Hugging Face `svhn` | 73,257 train / 26,032 test / 531,131 extra (~1.1 GB local) | Image classification, OOD pair | `datasets/svhn/` | Common OOD counterpart |

See `datasets/README.md` for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| RandOpt (Neural Thickets) | https://github.com/sunrainyg/RandOpt | Neural thicket expert generation from pretrained weights | `code/neural-thickets-randopt/` | Directly aligned with hypothesis |
| ens-div-ood-detect | https://github.com/Guoxoug/ens-div-ood-detect | Diversity analysis for deep ensemble OOD detection | `code/ens-div-ood-detect/` | Includes reproducible scripts |
| HQ-Ensemble | https://github.com/git-disl/HQ-Ensemble | Hierarchical ensemble pruning | `code/hq-ensemble/` | Compute-efficient ensemble subset selection |

See `code/README.md` for details.

## Resource Gathering Notes

### Search Strategy
- Used paper-finder diligent search first.
- Used exact-title arXiv retrieval for high-relevance papers.
- Parsed paper/repo links from downloaded PDFs and cloned relevant code.

### Selection Criteria
- Direct relevance to adversarial + OOD robustness.
- Explicit ensemble diversity or perturbation mechanisms.
- Availability of PDFs and, where possible, open-source code.

### Challenges Encountered
- Semantic Scholar API returned rate limits (HTTP 429) during bulk metadata pulls.
- Some papers did not expose a direct open-access PDF in this environment.
- One duplicate set of PDFs exists from an early naming pass (kept to avoid destructive operations).

### Gaps and Workarounds
- `DENL (2023)` remained metadata-only (no accessible open PDF found here).
- Workaround: expanded core set with directly downloadable, method-relevant arXiv papers.

## Recommendations for Experiment Design

1. **Primary dataset(s)**: Start with CIFAR-10 (ID) + SVHN (OOD); optionally add CIFAR-10-C/TinyImageNet for broader shifts.
2. **Baseline methods**: Single model, deep ensemble, Ensemble Adversarial Training baseline, PEP perturbation ensemble, Neural Thickets/RandOpt.
3. **Evaluation metrics**: Clean accuracy, robust accuracy under PGD/AutoAttack, AUROC/AUPR/FPR95, ECE/NLL.
4. **Code to adapt/reuse**: `neural-thickets-randopt` for expert generation; `ens-div-ood-detect` for diversity/OOD tooling; `hq-ensemble` for pruning studies.

## Research Execution Log (2026-03-23)

### What Was Executed
- Implemented full pipeline in `src/run_research.py` for:
  - CIFAR-10 base model training (ResNet-18 adapted for 32x32)
  - perturbation-based ensembles (`random`, `orthogonal`, `adversarial` directions)
  - clean accuracy/calibration evaluation
  - PGD robustness evaluation (`eps` = 2/255, 4/255, 8/255)
  - OOD detection against SVHN via max-softmax score
  - diversity metrics + bootstrap statistical tests

### Key Artifacts Generated
- `results/summary_table.csv`
- `results/metrics.json`
- `results/statistical_analysis.json`
- `results/environment.json`
- `results/train_history.csv`
- plots in `results/plots/`

### Notes
- Initial run surfaced a schema mismatch (`img` vs `image` column) for SVHN; fixed by adding automatic image-key detection in dataset wrapper.
- Added checkpoint-resume logic to avoid retraining when `results/models/base_model.pt` already exists.
