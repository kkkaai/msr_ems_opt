# Layer-Coupled Automated Hyperparameter Tuning for Hierarchical MSR

This repository contains the code used for the paper:

**Layer-Coupled Automated Hyperparameter Tuning for Hierarchical Multi-Superquadric Recovery**

The code focuses on hierarchical EMS-DBSCAN multi-superquadric recovery, automated layer-wise hyperparameter search, noise robustness evaluation, and geometry-level grasp candidate generation.

## Repository Scope

The open-source branch intentionally includes only the code and lightweight metadata needed to reproduce the experiments. It does not include large datasets, generated reconstruction outputs, paper source files, review material, or local caches.

Included components:

- `src/`: core hierarchical EMS-DBSCAN runners and shared utilities.
- `scripts/`: data conversion, batch execution, labeling, and rendering helpers.
- `exp1/`: single-superquadric recovery benchmark scripts.
- `exp2_3/`: KIT-multi recovery and budget-analysis scripts.
- `exp4_noise/`: quantitative noisy-input robustness evaluation.
- `exp5/grasp_generation_study/`: pregrasp-based two-finger and folding-hand grasp candidate generation.
- `data/kit_superquadric_labels.csv`: lightweight object-selection labels used by the experiment scripts.
- `hand/folding_hand_right/`: right folding-hand MJCF/URDF and meshes used by the grasp candidate generator.

Excluded components:

- KIT object point clouds and meshes.
- Generated experiment outputs under `results/`.
- Overleaf/LaTeX source and reviewer-response working files.
- Local reference PDFs and private notes.

## Requirements

Create a Python environment and install the Python dependencies:

```bash
pip install -r requirements.txt
```

The recovery scripts also require the public EMS superquadric fitting implementation used by the original EMS baseline. Place or install it under:

```text
external/EMS-superquadric_fitting/
```

The expected Python path is:

```text
external/EMS-superquadric_fitting/Python/src
```

See `THIRD_PARTY_NOTICES.md` for third-party attribution.

## Data Preparation

The scripts expect KIT point clouds in:

```text
data/KIT_ObjectModels_25k_ply/
```

If starting from OBJ files, use:

```bash
python scripts/obj_to_ply_pointcloud.py \
  --input-root data/KIT_ObjectModels_25k_obj \
  --output-root data/KIT_ObjectModels_25k_ply
```

Large data files are intentionally ignored by git.

## Main Experiments

Single-superquadric benchmark:

```bash
python exp1/run_exp1_single.py \
  --labels data/kit_superquadric_labels.csv \
  --ply-root data/KIT_ObjectModels_25k_ply \
  --output exp1/results/e1_single \
  --methods grid,pso,cmaes \
  --seeds 0,1,2
```

KIT-multi recovery:

```bash
python exp2_3/exp2_multi/run_exp2_multi.py \
  --labels data/kit_superquadric_labels.csv \
  --ply-root data/KIT_ObjectModels_25k_ply \
  --output exp2_3/results/e2_multi \
  --methods grid,pso,cmaes \
  --seeds 0,1,2
```

Noise robustness:

```bash
python exp4_noise/run_noise_robustness.py \
  --stage all \
  --profile 5x4 \
  --methods grid,pso,cmaes \
  --sigmas 0.005,0.01 \
  --seeds 0
```

Pregrasp-based grasp candidate generation:

```bash
python exp5/grasp_generation_study/run_generate_grasps.py --num-grasps 5

python exp5/grasp_generation_study/run_generate_folding_hand_grasps.py \
  --num-grasps 5 \
  --max-candidates 8 \
  --maxiter 35 \
  --skip-render
```

## Notes

- The grasp validity values produced by the grasp scripts are geometry-level filters, not physical MuJoCo success rates.
- The paper's full experimental outputs are not committed to this branch to keep the repository lightweight.
- Generated outputs should be written under experiment-specific `results/` folders, which are ignored by git.
