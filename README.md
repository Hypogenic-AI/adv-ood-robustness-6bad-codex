# Neural Thicket Robustness Stress Test

This project evaluates whether ensembles built from weight perturbations around a pretrained CIFAR-10 model improve adversarial and OOD robustness. It compares `single`, `random`, `orthogonal`, and `adversarial-direction` perturbation ensembles under identical compute and ensemble-size budgets.

## Key Findings
- Random perturbation ensemble improved PGD robustness at `eps=8/255` vs single model (`+0.45` points, bootstrap CI excludes 0).
- Orthogonal perturbations were statistically indistinguishable from random on key robustness metrics.
- Adversarial-direction perturbations greatly increased disagreement but degraded clean performance and did not improve robustness.
- All non-adversarially-trained methods remained highly vulnerable under strong PGD attacks.

## Reproduce
1. Activate environment:
```bash
source .venv/bin/activate
```
2. Run experiment pipeline:
```bash
python src/run_research.py
```
3. Inspect outputs:
- `results/summary_table.csv`
- `results/metrics.json`
- `results/statistical_analysis.json`
- `results/plots/*.png`

## File Structure
- `planning.md`: experiment design and motivation/novelty assessment.
- `src/run_research.py`: full training/evaluation pipeline.
- `results/`: metrics, model checkpoint, runtime, and plots.
- `REPORT.md`: full research report with interpretation and limitations.
- `literature_review.md`, `resources.md`: pre-gathered context and resources.

Full details: see `REPORT.md`.
