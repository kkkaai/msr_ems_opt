# Quantitative Noise Robustness Experiment

This experiment evaluates the eight KIT objects used in the existing qualitative comparison:

- CokePlasticLarge
- Deodorant
- FizzyTabletsCalcium
- HamburgerSauce
- Heart1
- MelforBottle
- Moon
- Slotted_Screwdriver

## Final protocol

- Noise model:
  \(\widetilde{\mathbf p}=\mathbf p+\eta\),
  \(\eta\sim\mathcal N(0,\sigma_n^2 I)\).
- The per-axis noise standard deviation is defined as a fraction of the clean
  point-cloud bounding-box diagonal. Thus, `0.005` and `0.01` correspond to
  0.5% and 1% of the object diagonal, respectively.
- New noise levels: `0.005` and `0.01`.
- Clean condition: reused from
  `exp2_3/results/e2_multi_5x4_f005_c001`.
- Primary methods: Grid, PSO, and CMA-ES.
- Primary seed: `0`, matching the displayed Figure 4 reconstructions.
- Matched `5x4` optimizer settings:
  - Grid: `5^4=625` candidates per active layer.
  - PSO: swarm size 25, maximum 25 iterations.
  - CMA-ES: population size 25, maximum 25 iterations.
- Objective:
  `distance_fit + 0.05 * (1 - coverage) + 0.001 * N_sq`.
- Other EMS and hierarchy settings are identical to Figure 4.

The main run contains:

`8 objects x 2 noise levels x 3 methods = 48 reconstructions`.

The script supports additional noise/optimizer seeds. For stronger confidence
analysis, run `--seeds 0,1,2`; each seed creates a shared noisy point cloud used
by all three methods.

## Metrics

The optimization objective is computed against the noisy input. Independent
evaluation metrics are computed against the original clean point cloud. All
evaluation distances are divided by the clean point-cloud bounding-box diagonal:

- bidirectional clean-reference Chamfer-L1;
- clean-to-reconstruction and reconstruction-to-clean mean distances;
- 95th-percentile directional distances;
- precision, recall, and F-score at normalized thresholds 0.01 and 0.02;
- degradation relative to the clean `5x4` reconstruction;
- number of superquadrics;
- runtime and the original noisy-input objective components.

The final reconstructed superquadric surfaces are sampled by constructing a
triangular parametric mesh and sampling triangles in proportion to surface area.

Each reconstruction JSON also stores `primitive_assignments`. For every
superquadric, this records its global `primitive_id`, hierarchy level, source
segment, and the original input-row indices classified as EMS inliers. The
noisy PLY generator preserves point order, so these indices address the
corresponding points in both the noisy input and the original clean cloud.
Index propagation is bookkeeping only: indices are never passed to EMS,
DBSCAN, the optimizer, or the objective.

The evaluator additionally reports `inlier_chamfer_l1` and
`inlier_fscore_*`. These compare each primitive only with the clean points
identified by that primitive's recorded EMS inlier indices. The original
whole-cloud Chamfer and F-score remain in the output as coverage-sensitive
metrics.

## Outputs

The script writes:

- `pointclouds/`: deterministic noisy PLY files;
- `runs/`: reconstruction JSON files;
- `logs/`: stdout/stderr for each reconstruction;
- `metrics_raw.csv`;
- `metrics_with_degradation.csv`;
- `metrics_summary.csv`;
- `figures/noise_metrics_summary.png` and `.pdf`;
- per-object figures:
  - clean point cloud;
  - noisy point cloud;
  - reconstruction only;
  - reconstruction overlaid with the noisy input.

## Commands

Generate noisy point clouds:

```bash
conda run --no-capture-output -n msr_ems_opt python -u exp4_noise/run_noise_robustness.py \
  --stage generate
```

Smoke test:

```bash
conda run --no-capture-output -n msr_ems_opt python -u exp4_noise/run_noise_robustness.py \
  --stage all \
  --profile smoke \
  --objects CokePlasticLarge_25k \
  --sigmas 0.005 \
  --seeds 0 \
  --output exp4_noise/results/smoke
```

Full indexed `5x4` experiment, run as three separate reconstruction groups.
Use a new output directory because existing JSON files do not contain point
indices.

```bash
conda run --no-capture-output -n msr_ems_opt python -u exp4_noise/run_noise_robustness.py \
  --stage generate \
  --profile 5x4 \
  --objects CokePlasticLarge_25k,Deodorant_25k,FizzyTabletsCalcium_25k,HamburgerSauce_25k,Heart1_25k,MelforBottle_25k,Moon_25k,Slotted_Screwdriver_25k \
  --sigmas 0.005,0.01 \
  --seeds 0 \
  --output exp4_noise/results/noise_8objects_5x4_indexed
```

PSO:

```bash
conda run --no-capture-output -n msr_ems_opt python -u exp4_noise/run_noise_robustness.py \
  --stage run --profile 5x4 --sigmas 0.005,0.01 --seeds 0 \
  --methods pso \
  --output exp4_noise/results/noise_8objects_5x4_indexed
```

CMA-ES:

```bash
conda run --no-capture-output -n msr_ems_opt python -u exp4_noise/run_noise_robustness.py \
  --stage run --profile 5x4 --sigmas 0.005,0.01 --seeds 0 \
  --methods cmaes \
  --output exp4_noise/results/noise_8objects_5x4_indexed
```

Grid:

```bash
conda run --no-capture-output -n msr_ems_opt python -u exp4_noise/run_noise_robustness.py \
  --stage run --profile 5x4 --sigmas 0.005,0.01 --seeds 0 \
  --methods grid \
  --output exp4_noise/results/noise_8objects_5x4_indexed
```

After all three groups finish, compute the metrics and figures:

```bash
conda run --no-capture-output -n msr_ems_opt python -u exp4_noise/run_noise_robustness.py \
  --stage evaluate --profile 5x4 --sigmas 0.005,0.01 --seeds 0 \
  --methods grid,pso,cmaes \
  --output exp4_noise/results/noise_8objects_5x4_indexed
```

Resume an interrupted run by executing the same command. Existing JSON files
are reused by default.

The command displays separate `tqdm` progress bars for noisy-point-cloud
generation, reconstruction, and metric/figure generation. During
reconstruction, the progress bar shows the current noise level, seed, method,
and object. One step is completed after the corresponding reconstruction JSON
has been written.

Recompute metrics and figures without rerunning reconstruction:

```bash
conda run --no-capture-output -n msr_ems_opt python -u exp4_noise/run_noise_robustness.py \
  --stage evaluate \
  --output exp4_noise/results/noise_8objects_5x4
```

## Publication-view image export

The automatic Matplotlib figures are intended for result inspection and metric
verification. For publication figures, use `interactive_render.py` to select
the view manually in a Mayavi window.

The recommended workflow is:

1. Open one representative point cloud or reconstruction.
2. Rotate and zoom the scene manually.
3. Press `s` to save the image and camera parameters.
4. Reuse the saved camera JSON for the other methods and noise levels.
5. Press `q` to close the window.

Example: choose and save a camera view:

```bash
conda run --no-capture-output -n msr_ems_opt python -u exp4_noise/interactive_render.py \
  --json exp4_noise/results/noise_8objects_5x4/runs/sigma_0p005/seed_0/pso/CokePlasticLarge_25k.json \
  --points exp4_noise/results/noise_8objects_5x4/pointclouds/sigma_0p005/seed_0/CokePlasticLarge_25k.ply \
  --mode with_points \
  --output exp4_noise/results/noise_8objects_5x4/manual_figures/CokePlasticLarge_pso_overlay.png \
  --camera-out exp4_noise/results/noise_8objects_5x4/manual_figures/CokePlasticLarge_camera.json
```

Reuse exactly the same view for CMA-ES:

```bash
conda run --no-capture-output -n msr_ems_opt python -u exp4_noise/interactive_render.py \
  --json exp4_noise/results/noise_8objects_5x4/runs/sigma_0p005/seed_0/cmaes/CokePlasticLarge_25k.json \
  --points exp4_noise/results/noise_8objects_5x4/pointclouds/sigma_0p005/seed_0/CokePlasticLarge_25k.ply \
  --mode with_points \
  --camera-in exp4_noise/results/noise_8objects_5x4/manual_figures/CokePlasticLarge_camera.json \
  --output exp4_noise/results/noise_8objects_5x4/manual_figures/CokePlasticLarge_cmaes_overlay.png
```
