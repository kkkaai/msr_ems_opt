# Exp1: Single Superquadric Benchmark (KIT Single)

This folder contains reproducible scripts for E1 comparing only:
- `grid`
- `pso`
- `cma-es`

No `fixed` baseline is used.

## 1) Run benchmark

From repo root:

```bash
conda run -n msr_ems_opt python exp1/run_exp1_single.py \
  --labels data/kit_superquadric_labels.csv \
  --ply-root data/KIT_ObjectModels_25k_ply \
  --output exp1/results/e1_single \
  --methods grid,pso,cmaes \
  --seeds 0,1,2
```

Notes:
- Current label file contains `single=85`.
- If `cma` is not installed, `cma-es` runs will fail with clear logs. Install with:
  - `conda run -n msr_ems_opt pip install cma`
- Default search budget is unified to `10x10`:
  - `grid`: `OutlierRatioSteps=10`, `SigmaSteps=10`
  - `pso`: `swarmsize=10`, `maxiter=10`
  - `cma-es`: `popsize=10`, `maxiter=10`
- The runner now prints run-level progress lines:
  - `[run 0001/0595] START/OK/FAIL object | method | seed`

## 2) Generate figures

```bash
conda run -n msr_ems_opt python exp1/plot_exp1.py \
  --exp-root exp1/results/e1_single \
  --methods grid,pso,cmaes
```

Outputs:
- `exp1/results/e1_single/table1_summary.csv`
- `exp1/results/e1_single/table1_summary.md`
- `exp1/results/e1_single/figures/fig1_boxplots.png`
- `exp1/results/e1_single/figures/fig2_heatmap.png`

## 3) Smoke test (optional)

```bash
conda run -n msr_ems_opt python exp1/run_exp1_single.py \
  --output exp1/results/e1_single_smoke \
  --methods grid,pso \
  --max-objects 3 \
  --seeds 0,1

conda run -n msr_ems_opt python exp1/plot_exp1.py \
  --exp-root exp1/results/e1_single_smoke \
  --methods grid,pso
```
