# Literature Review: Adversarial and OOD Robustness of Neural Thicket-Style Ensembles

## Review Scope

### Research Question
How robust are neural thicket-style ensembles (constructed from perturbations around pretrained weights) under adversarial attacks and OOD shifts, and what diversity mechanisms most improve resilience?

### Inclusion Criteria
- Ensemble-based robustness methods for neural networks.
- Adversarial robustness and/or OOD detection benchmarks.
- Explicit treatment of diversity, perturbation, pruning, or certification.
- Practical implementation details and reproducible setups.

### Exclusion Criteria
- Non-neural or unrelated ensemble methods.
- Domain-specific papers with no transferable robustness methodology.
- Papers without sufficient methodological detail.

### Time Frame
- Primarily 2016-2026, with emphasis on 2020-2026.

### Sources
- Paper-finder service (diligent mode)
- arXiv API/title search
- Semantic Scholar metadata/API
- GitHub repositories linked from papers

## Search Log

| Date | Query | Source | Results | Notes |
|------|-------|--------|---------|-------|
| 2026-03-23 | adversarial robustness out-of-distribution robustness neural network ensembles diversity orthogonal perturbations | paper-finder | 27 papers | Initial relevance-ranked set |
| 2026-03-23 | focused exact-title arXiv queries for top ensemble/OOD papers | arXiv API | 11 strong matches + Neural Thickets | Used for direct PDF retrieval |
| 2026-03-23 | neural thicket / perturbation ensemble robustness | arXiv API | mixed relevance | Retained most relevant works only |

## Screening Results

| Stage | Count | Outcome |
|------|------:|---------|
| Initial retrieval | 27 | Paper-finder output |
| Full-text accessible and downloaded | 12 core + 2 duplicate-named copies | Main review corpus |
| Deep-read subset | 4 | Neural Thickets, PEP, Joint Certification, Deep Ensemble Diversity for OOD |

## Research Area Overview

The literature converges on a common observation: ensembles improve both confidence calibration and robustness, but robustness gains vary strongly with **expert diversity** and **error independence**. Work from adversarial training emphasizes distributional coverage of attacks, while OOD work emphasizes score separation and uncertainty quality. Newer methods (including Neural Thickets/RandOpt) shift focus toward generating many nearby experts around pretrained models, making diversity generation more compute-efficient than full retraining.

## Key Papers

### 1) Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights (2026)
- **Authors**: Yulu Gan, Phillip Isola
- **Source**: arXiv (2603.12228)
- **Key Contribution**: Shows dense neighborhoods of useful task experts around pretrained weights and introduces RandOpt-style expert generation.
- **Methodology**: Randomized optimization/perturbation around pretrained checkpoints to sample multiple competent experts.
- **Datasets Used**: Multi-task/post-training benchmarks (see repo scripts/data docs).
- **Results**: Strong claim that many diverse high-performing experts are reachable with lightweight search.
- **Code Available**: Yes (`code/neural-thickets-randopt`).
- **Relevance**: Directly matches hypothesis foundation.

### 2) Robust Deep Learning Ensemble against Deception (2020)
- **Authors**: Wenqi Wei, Ling Liu
- **Source**: IEEE TDSC + arXiv (2009.06589)
- **Key Contribution**: Joint defense against adversarial and OOD inputs via diversity-driven verification (XEnsemble).
- **Methodology**: Input denoising verifiers + output disagreement/consensus checks.
- **Datasets Used**: Vision datasets with attack/OOD evaluations.
- **Results**: Improved defense success and OOD detection vs representative baselines.
- **Code Available**: Not directly resolved in this run.
- **Relevance**: Strongly aligned with dual-threat setting in hypothesis.

### 3) Ensemble Adversarial Training: Attacks and Defenses (2017)
- **Authors**: Tramèr et al.
- **Source**: ICLR 2018 / arXiv 1705.07204
- **Key Contribution**: Transfers adversarial examples from multiple static models to reduce gradient masking and improve robustness.
- **Methodology**: Adversarial training with ensemble-transferred perturbations.
- **Datasets Used**: Image classification benchmarks (e.g., MNIST/ImageNet variants in original work).
- **Results**: Better black-box robustness than single-model adversarial training baselines.
- **Relevance**: Canonical baseline for attack-aware ensemble diversity.

### 4) Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles (2016)
- **Authors**: Lakshminarayanan et al.
- **Source**: arXiv 1612.01474
- **Key Contribution**: Strong uncertainty and calibration baseline using independently trained models.
- **Methodology**: Multiple random initializations + proper scoring rules.
- **Datasets Used**: Vision/regression benchmarks.
- **Results**: Competitive uncertainty quality without Bayesian complexity.
- **Relevance**: Baseline for robustness/OOD confidence estimation.

### 5) Out-of-Distribution Detection Using an Ensemble of Self Supervised Leave-out Classifiers (2018)
- **Authors**: Vyas et al.
- **Source**: arXiv 1809.03576
- **Key Contribution**: Ensemble OOD detector using leave-out partitions and self-supervised signal.
- **Methodology**: Partitioned classifiers trained to reject held-out subsets.
- **Datasets Used**: Common vision OOD protocols.
- **Results**: Improved OOD separation over simpler confidence-based baselines.
- **Relevance**: OOD-focused ensemble design baseline.

### 6) Enhancing Certifiable Robustness via a Deep Model Ensemble (2019)
- **Authors**: Zhang et al.
- **Source**: arXiv 1910.14655
- **Key Contribution**: Certification-aware ensemble weighting.
- **Methodology**: Optimize ensemble weights under certified bounds.
- **Results**: Better certified guarantees than naive averaging.
- **Relevance**: Critical for provable robustness angle.

### 7) Certifying Joint Adversarial Robustness for Model Ensembles (2020)
- **Authors**: Ahmad, Evans
- **Source**: arXiv 2004.10250
- **Key Contribution**: Formal framework for certifying joint ensemble robustness.
- **Methodology**: Robustness certificates that account for ensemble aggregation behavior.
- **Results**: Joint guarantees can exceed component-level guarantees in useful regimes.
- **Relevance**: Supports robust evaluation protocol design.

### 8) PEP: Parameter Ensembling by Perturbation (2020)
- **Authors**: Mehrtash et al.
- **Source**: arXiv 2010.12721
- **Key Contribution**: Build ensembles through parameter perturbations rather than full retraining.
- **Methodology**: Perturb and aggregate model parameters around a reference model.
- **Results**: Efficient ensembles with competitive predictive performance.
- **Relevance**: Direct precursor to perturbation-driven thicket construction.

### 9) Sequential Bayesian Neural Subnetwork Ensembles (2022)
- **Authors**: Jantre et al.
- **Source**: arXiv 2206.00794
- **Key Contribution**: Bayesian sequential subnetwork ensemble approach for uncertainty and efficiency.
- **Methodology**: Subnetwork sampling with Bayesian update structure.
- **Relevance**: Alternative to full-model ensembles under resource constraints.

### 10) Batch-Ensemble Stochastic Neural Networks for OOD Detection (2022)
- **Authors**: Chen et al.
- **Source**: arXiv 2206.12911
- **Key Contribution**: Parameter-sharing stochastic ensemble design targeting OOD detection.
- **Relevance**: Useful low-cost ensemble baseline.

### 11) On the Usefulness of Deep Ensemble Diversity for OOD Detection (2022)
- **Authors**: Xia, Bouganis
- **Source**: arXiv 2207.07517
- **Key Contribution**: Explicitly studies relation between diversity and OOD performance.
- **Methodology**: Empirical analysis of ensemble diversity metrics vs OOD detection metrics.
- **Code Available**: Yes (`code/ens-div-ood-detect`).
- **Relevance**: Directly supports hypothesis statement on diversity dependence.

### 12) Hierarchical Pruning of Deep Ensembles with Focal Diversity (2023)
- **Authors**: Wu et al.
- **Source**: arXiv 2311.10293
- **Key Contribution**: Pruning ensembles while preserving diversity and robustness benefits.
- **Code Available**: Related tooling in `code/hq-ensemble` family.
- **Relevance**: Important for scaling thickets under compute limits.

## Common Methodologies

- **Independent deep ensembles**: multiple independently trained models for uncertainty and robustness.
- **Perturbation-based ensembling**: experts sampled via parameter perturbations around pretrained weights (PEP, Neural Thickets).
- **Adversarially informed training**: ensemble adversarial training and transferred attacks.
- **Diversity-aware selection/pruning**: hierarchical/focal diversity methods.
- **Certification-aware aggregation**: robust bound optimization and joint certificates.

## Standard Baselines

- Single pretrained model (no ensemble).
- Deep ensemble with random seeds.
- Ensemble adversarial training baseline.
- Parameter-sharing ensemble variants (BatchEnsemble-style).
- Diversity-pruned subset selection from a larger expert pool.

## Evaluation Metrics

- **Clean accuracy** on in-distribution test data.
- **Adversarial robustness**: robust accuracy under PGD/AutoAttack-style threat models.
- **OOD detection**: AUROC, AUPR, FPR95 (and optionally detection error).
- **Calibration/uncertainty**: ECE, NLL, Brier score.
- **Diversity metrics**: disagreement, Q-statistics, pairwise error correlation, representation diversity.

## Datasets in the Literature

- CIFAR-10 / CIFAR-100 (core ID benchmarks).
- SVHN, TinyImageNet, LSUN, iSUN, Textures, OpenImage-O (common OOD sets).
- ImageNet-200 and near-OOD suites for larger-scale evaluation.

## Gaps and Opportunities

1. Limited direct adversarial+OOD co-evaluation for perturbation-thicket methods.
2. Diversity metrics are often measured post hoc, not optimized during expert sampling.
3. Certification analyses lag behind rapidly evolving perturbation-based ensemble methods.
4. Cost-aware scaling laws (experts vs robustness gain) remain undercharacterized.

## Recommendations for Our Experiment

- **Recommended datasets**:
  - ID: CIFAR-10 (downloaded).
  - OOD: SVHN (downloaded), plus optional CIFAR-10-C/TinyImageNet for richer shifts.
- **Recommended baselines**:
  - Single model.
  - Standard deep ensemble.
  - PEP-style perturbation ensemble.
  - RandOpt/Neural Thickets expert sampling.
- **Recommended metrics**:
  - Clean accuracy, robust accuracy (PGD/AutoAttack), AUROC/AUPR/FPR95, ECE/NLL.
- **Methodological considerations**:
  - Control ensemble size and total compute budget.
  - Measure diversity explicitly and correlate with robustness metrics.
  - Evaluate both random perturbations and constrained directions (orthogonal/adversarial directions).
  - Include calibration and uncertainty diagnostics, not just top-1 accuracy.
