# Pregrasp-Based Grasp Generator

This folder contains a runnable grasp-generation baseline for the eight scaled Figure 4 reconstructions.

The current implementation uses a lightweight analytic parallel-jaw hand because the real dexterous-hand FK model is not stored in this repository. The code is organized so that the hand model can later be replaced by a URDF/MJCF-based dexterous hand adapter while keeping the same superquadric geometry, pregrasp initialization, optimization, and reporting pipeline.

## Run

```bash
python exp5/grasp_generation_study/run_generate_grasps.py --num-grasps 5 --max-candidates 30 --maxiter 180
```

The script reads the uniformly scaled 20 cm models from:

```text
exp5/grasp_generation_study/results/figure4_numbered_cmaes/scaled_json_20cm/
```

and writes:

```text
exp5/grasp_generation_study/results/pregrasp_grasps/selected_grasps.json
exp5/grasp_generation_study/results/pregrasp_grasps/selected_grasps_metrics.csv
exp5/grasp_generation_study/results/pregrasp_grasps/visualizations/
```

## Pregrasp Initialization

Each optimization starts from a pregrasp pose near a target superquadric rather than from a zero pose. For each target SQ, the script:

1. samples approach directions from the target SQ principal axes;
2. chooses a closing direction orthogonal to the approach direction;
3. estimates two opposing surface anchors on the target SQ;
4. places the open hand outside the target region by a fixed pregrasp offset;
5. runs local L-BFGS-B optimization from this pregrasp.

## Objective

The simplified objective optimizes a local pose and gripper width:

```text
E = 100 E_tip_contact + 15 E_aux_contact + 100 E_pen + E_width + E_pose_drift
```

The terms encourage fingertip contact with the selected target SQ, keep auxiliary hand points close to the object, penalize object penetration using sampled surface distance, and keep the optimized pose close to the pregrasp.

## Notes

The reported `success` field is a geometric filter for this analytic baseline, not a physical simulation success label. MuJoCo validation and the real dexterous-hand FK adapter should be added before using these values as final quantitative grasp results in the paper.

## Folding Hand Right

The repository also contains a kinematic candidate generator for the real right folding hand model:

```bash
python exp5/grasp_generation_study/run_generate_folding_hand_grasps.py --num-grasps 5 --max-candidates 8 --maxiter 35 --skip-render
```

It uses:

```text
hand/folding_hand_right/folding_hand_right.xml
```

and writes:

```text
exp5/grasp_generation_study/results/folding_hand_right_pregrasp_grasps/selected_folding_hand_grasps.json
exp5/grasp_generation_study/results/folding_hand_right_pregrasp_grasps/selected_folding_hand_metrics.csv
```

The generator starts from a pregrasp pose by aligning the open-hand fingertip centroid, computed from the folding-hand forward kinematics, with the selected target superquadric region. It then locally optimizes the wrist pose and the seven actuator-style controls:

```text
plam_th, th, plam_other, ff, mf, rf, lf
```

The current `success` flag remains a conservative geometry-only filter. Since `mujoco` is available in the base environment and the XML loads successfully, the next step is to replay these saved poses in MuJoCo and report physical grasp stability separately.
