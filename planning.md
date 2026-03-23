# Research Plan: Probing the Limits of Neural Thicket Ensembles

## Motivation & Novelty Assessment

### Why This Research Matters
Adversarial and OOD failures remain deployment blockers for image classifiers, especially when safety depends on calibrated uncertainty. Perturbation-based ensembles are attractive because they can be built cheaply around a pretrained model, but their robustness under strong stress tests is not well characterized. Establishing when these ensembles help, and when they fail, can guide robust model selection for security-sensitive applications.

### Gap in Existing Work
From `literature_review.md`, prior work confirms that ensemble diversity correlates with uncertainty quality, but direct *joint* evaluation of perturbation-based thickets under both adversarial attack and OOD shift is limited. Diversity is often measured after training rather than explicitly imposed during expert construction. There is also limited evidence comparing random perturbations to structured perturbation directions under equal ensemble size and compute.

### Our Novel Contribution
We test a controlled robustness benchmark for thicket-style ensembles built around one pretrained CIFAR-10 model, comparing three expert-generation strategies under identical budgets: random Gaussian perturbations, orthogonalized perturbations, and adversarially selected (loss-increasing) perturbations. We connect diversity diagnostics to robustness outcomes, quantifying whether explicit direction design improves resilience.

### Experiment Justification
- Experiment 1: Single model vs random perturbation ensemble. Needed to verify whether thicket-style sampling alone yields robustness/OOD gains over a single checkpoint.
- Experiment 2: Random vs orthogonal perturbation directions. Needed to test whether stronger expert independence improves adversarial and OOD resilience.
- Experiment 3: Random vs adversarially selected directions. Needed to test whether stress-aware perturbation directions provide better robustness than purely random sampling.
- Experiment 4: Diversity-robustness correlation analysis. Needed to validate the mechanism in the hypothesis (diversity/independence as the driver).

## Research Question
Do neural thicket ensembles formed by perturbing a pretrained CIFAR-10 model improve adversarial and OOD robustness, and do explicit diversity-inducing perturbation directions (orthogonal or adversarial) improve robustness beyond random perturbations?

## Background and Motivation
Neural thickets and perturbation ensembling suggest that useful experts can be sampled around a strong pretrained solution at lower cost than full independent training. Deep ensemble/OOD literature indicates diversity is central to uncertainty quality, while adversarial robustness literature emphasizes transfer and gradient diversity. This project bridges those threads with one controlled benchmark: same base model, same ensemble size, different perturbation direction strategies.

## Hypothesis Decomposition
- H1: Any perturbation ensemble (random directions) improves robustness metrics vs single model at similar clean accuracy.
- H2: Orthogonal perturbation directions produce higher expert diversity than random perturbations.
- H3: Adversarially selected perturbation directions increase adversarial robustness relative to random perturbations.
- H4: Diversity metrics (pairwise disagreement, prediction correlation) positively correlate with OOD AUROC and robust accuracy.

Independent variables:
- Ensemble construction method: `single`, `random`, `orthogonal`, `adversarial-direction`.
- Perturbation scale `sigma` and ensemble size `M`.
- Attack strength `epsilon` for PGD.

Dependent variables:
- Clean accuracy, PGD robust accuracy, OOD AUROC/AUPR/FPR95, ECE/NLL.
- Diversity metrics: pairwise disagreement, mean error correlation.

Success criteria:
- At least one structured diversity method outperforms random perturbation ensemble on robust accuracy and/or OOD AUROC with non-overlapping bootstrap CI of improvement.

Alternative explanations to check:
- Gains may be due to ensembling itself, not diversity.
- Gains may come from confidence smoothing while accuracy drops.
- Perturbation magnitude may dominate direction strategy.

## Proposed Methodology

### Approach
Train one competent CIFAR-10 base model, then generate multiple experts by weight-space perturbations around that checkpoint. Compare three perturbation direction strategies under matched ensemble size and perturbation norm. Evaluate ID performance, adversarial robustness (PGD), OOD detection (SVHN), calibration, and diversity diagnostics.

### Experimental Steps
1. Environment and reproducibility setup: fixed seeds, logging, GPU/mixed precision, version capture.
2. Data loading: CIFAR-10 train/test as ID; SVHN test as OOD; data quality checks and class counts.
3. Train baseline model on CIFAR-10 (ResNet-18 adapted to 32x32).
4. Build ensembles:
   - Random Gaussian direction thicket.
   - Orthogonalized direction thicket.
   - Adversarial direction thicket (weight directions from loss gradients on calibration minibatches).
5. Recalibrate BatchNorm statistics for each perturbed expert using CIFAR-10 train subset.
6. Evaluate methods on clean test accuracy and calibration.
7. Run PGD attacks (`L_inf`, eps in {2/255, 4/255, 8/255}, 10-20 steps) and report robust accuracy.
8. Evaluate OOD detection (ID CIFAR-10 test vs OOD SVHN test) with MSP and entropy scores.
9. Compute diversity metrics and correlate with robustness/OOD metrics.
10. Statistical analysis via bootstrap confidence intervals and hypothesis tests.

### Baselines
- Single pretrained model (no ensemble).
- Random perturbation ensemble (PEP/thicket-style baseline).
- Structured variants: orthogonal and adversarial-direction perturbation ensembles (proposed comparisons).

### Evaluation Metrics
- Classification: top-1 accuracy, negative log-likelihood (NLL).
- Calibration: Expected Calibration Error (ECE, 15 bins).
- Adversarial: robust accuracy under PGD at multiple eps.
- OOD: AUROC, AUPR, FPR95 (ID positive) from max-softmax confidence.
- Diversity: pairwise disagreement rate, pairwise error correlation.

### Statistical Analysis Plan
- Primary significance level: `alpha = 0.05`.
- For method comparisons on accuracy/robust accuracy: paired bootstrap over test examples (10,000 resamples) with 95% CI for difference.
- For OOD AUROC/AUPR/FPR95: bootstrap CI (stratified over ID/OOD samples).
- For diversity-performance relation: Spearman correlation with confidence intervals via bootstrap.
- Multiple comparisons: Holm correction over primary pairwise method tests.

## Expected Outcomes
Support for hypothesis if:
- Random perturbation ensembles improve robust/OOD metrics over single model.
- Orthogonal or adversarial-direction ensembles show higher diversity and better robustness/OOD than random perturbation ensembles.

Refutation if:
- Structured direction methods fail to improve robustness or degrade clean/OOD substantially.
- Diversity metrics do not correlate with robustness/OOD outcomes.

## Timeline and Milestones
- Phase A (20 min): finalize planning + checks.
- Phase B (20 min): environment/package/data validation.
- Phase C (60 min): implement training/evaluation pipeline.
- Phase D (60 min): run experiments and collect outputs.
- Phase E (40 min): analysis, plots, statistical tests.
- Phase F (30 min): documentation and reproducibility checks.
- Buffer (30% integrated): debugging and reruns.

## Potential Challenges
- Perturbations may collapse clean accuracy: mitigate with sigma sweep and BN recalibration.
- PGD runtime could be high: use subsampled robust set if needed and document it.
- OOD mismatch (digit-heavy SVHN): interpret AUROC with caveats and include calibration diagnostics.
- Single-base-model dependence: run multiple perturbation seeds and report variance.

## Success Criteria
- Complete reproducible pipeline with saved configs, metrics, and plots.
- Report clean/adversarial/OOD/calibration/diversity metrics for all methods.
- Provide statistically grounded comparison and mechanism-focused interpretation.
- Deliver `REPORT.md` with actual measured results and limitations.
