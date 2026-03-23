# Cloned Repositories

## Repo 1: neural-thickets-randopt
- URL: https://github.com/sunrainyg/RandOpt
- Purpose: Official codebase for *Neural Thickets* / RandOpt around pretrained weights.
- Location: `code/neural-thickets-randopt/`
- Key files:
  - `scripts/local_run.sh`
  - `scripts/single_node.sh`
  - `scripts/multiple_nodes.sh`
  - `baselines/`
  - `distillation/README.md`
- Notes:
  - Directly aligned with the project hypothesis.
  - Supports local and Slurm workflows.
  - Requires substantial compute for large-model post-training experiments.

## Repo 2: ens-div-ood-detect
- URL: https://github.com/Guoxoug/ens-div-ood-detect
- Purpose: Reproduces analysis from *On the Usefulness of Deep Ensemble Diversity for OOD Detection*.
- Location: `code/ens-div-ood-detect/`
- Key files:
  - `experiment_scripts/resnet50_imagenet200.sh`
  - `experiment_scripts/resnet50_imagenet200_from_logits.sh`
  - `experiment_scripts/present_results.sh`
  - `experiment_configs/change_paths.py`
- Notes:
  - Strong resource for diversity-vs-OOD measurement tooling.
  - Focuses on ImageNet-200 style settings; dataset path configuration required.

## Repo 3: hq-ensemble
- URL: https://github.com/git-disl/HQ-Ensemble
- Purpose: Hierarchical ensemble pruning and diversity-aware selection.
- Location: `code/hq-ensemble/`
- Key files:
  - `HQ-Ensemble.py`
  - `baselineDiversityPruning.py`
  - `env.sh`
- Notes:
  - Useful for compute-efficient ensemble subset selection.
  - Input preparation expects prediction vectors and labels by dataset.

## Quick Validation Performed

- All repositories cloned successfully.
- README files reviewed for requirements and entry points.
- No full training runs executed here due runtime/compute cost.
