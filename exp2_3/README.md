# exp2_3 实验脚本说明（E2 + E3）

本目录用于多超二次型重建实验：
- `exp2_multi/`：E2 主实验（`grid/pso/cmaes`，4 参数：`OutlierRatio, Sigma, Eps, MinPoints`）
- `exp3_budget/`：E3 预算公平实验（公平预算 vs 最佳精度模式）

## 1. E2 主实验

### 运行实验（可分方法跑）

```bash
conda run -n msr_ems_opt python exp2_3/exp2_multi/run_exp2_multi.py \
  --labels data/kit_superquadric_labels.csv \
  --ply-root data/KIT_ObjectModels_25k_ply \
  --output exp2_3/results/e2_multi \
  --methods grid \
  --seeds 0,1,2 \
  --FitnessMode distance_coverage_outlier_complexity \
  --LambdaCov 0.03 --LambdaOut 0.03 --LambdaComp 0.03
```

再分别运行：
- `--methods pso`
- `--methods cmaes`

### 可选：保存每个物体可视化截图

```bash
conda run -n msr_ems_opt python exp2_3/exp2_multi/run_exp2_multi.py \
  --output exp2_3/results/e2_multi \
  --methods pso \
  --save-images --image-seed 0 --image-format png --visualizeMode with_points
```

### 生成图表

```bash
conda run -n msr_ems_opt python exp2_3/exp2_multi/plot_exp2.py \
  --exp-root exp2_3/results/e2_multi \
  --methods grid,pso,cmaes \
  --format png \
  --reference-method pso \
  --image-seed 0
```

输出：
- `table2_summary.csv/.md`
- `figures/fig3_representative.*`
- `figures/fig4_pareto.*`

## 2. E3 预算公平实验

### 公平预算模式（推荐先跑）

```bash
conda run -n msr_ems_opt python exp2_3/exp3_budget/run_exp3_budget.py \
  --labels data/kit_superquadric_labels.csv \
  --ply-root data/KIT_ObjectModels_25k_ply \
  --output exp2_3/results/e3_budget_fair \
  --methods grid,pso,cmaes \
  --seeds 0,1,2 \
  --mode fair \
  --budget-evals-per-layer 625
```

### 最佳精度模式（补充实验）

```bash
conda run -n msr_ems_opt python exp2_3/exp3_budget/run_exp3_budget.py \
  --output exp2_3/results/e3_budget_best \
  --methods grid,pso,cmaes \
  --seeds 0,1,2 \
  --mode best
```

### 生成收敛图

```bash
conda run -n msr_ems_opt python exp2_3/exp3_budget/plot_exp3.py \
  --exp-root exp2_3/results/e3_budget_fair \
  --methods grid,pso,cmaes \
  --format png
```

输出：
- `table3_summary.csv/.md`
- `curves.csv`（收敛曲线原始数据）
- `figures/fig5_convergence.*`

### 不重跑重建：复用 E2 结果做 E3 补充分析

```bash
conda run -n msr_ems_opt python exp2_3/exp3_budget/reuse_e2_results_for_e3.py \
  --e2-root exp2_3/results/e2_multi_full \
  --output exp2_3/results/e3_from_e2_supp \
  --methods grid,pso,cmaes \
  --seeds 0,1,2
```

再画收敛图：

```bash
conda run -n msr_ems_opt python exp2_3/exp3_budget/plot_exp3.py \
  --exp-root exp2_3/results/e3_from_e2_supp \
  --methods grid,pso,cmaes \
  --format svg
```

## 3. 结果文件结构

每次 run 都会保存：
- `json/<method>/<object>__seed*.json`：完整重建结果（含总时间、分层时间、fitness 各项）
- `log/<method>/<object>__seed*.log`：命令行日志
- `runs_raw.csv`：逐次运行记录
- `per_object.csv`：按物体聚合（种子取中位数）

E2 额外保存：
- `table2_summary.csv/.md`

E3 额外保存：
- `table3_summary.csv/.md`
- `curves.csv`
