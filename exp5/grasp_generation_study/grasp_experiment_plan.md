# Quantitative Grasp Generation Experiment Plan

## 1. Scope and fixed reconstruction inputs

The experiment uses the eight objects shown in Figure 4:

1. HamburgerSauce
2. Heart1
3. Slotted_Screwdriver
4. FizzyTabletsCalcium
5. MelforBottle
6. Deodorant
7. CokePlasticLarge
8. Moon

For a reproducible and uniform protocol, the recommended reconstruction input is the **CMA-ES, seed 0, 5x4** result for every object. This avoids selecting a different optimizer or seed after inspecting grasp outcomes. If the paper later decides to use PSO instead, the preparation script can be changed by replacing the JSON root; the remaining protocol is unchanged.

Each superquadric is numbered in JSON order as `SQ1`, `SQ2`, etc. The numbered images and parameter map are stored under:

`exp5/grasp_generation_study/results/figure4_numbered_cmaes/`

## 2. Model scaling and coordinate convention

The recovered models are stored in millimeters. For grasp generation, each complete multi-superquadric model is uniformly scaled so that the maximum side of its reconstructed axis-aligned bounding box is 0.20 m:

```text
x_grasp = s (x_source - c)
s = 0.20 / max(AABB_extent_source)
```

where `c` is the center of the reconstructed AABB. The same uniform scale and translation are applied to:

- every superquadric scale parameter;
- every superquadric translation;
- the reconstructed surface used for collision checking;
- any visualization or simulation mesh.

Uniform scaling preserves shape, orientation, and relative part placement. Independent scaling along each axis must not be used.

### Scaled reconstruction dimensions

| Object | SQs | Source reconstructed AABB (mm) | Scaled AABB (m) |
|---|---:|---:|---:|
| HamburgerSauce | 4 | 59.97 x 176.54 x 82.86 | 0.068 x 0.200 x 0.094 |
| Heart1 | 1 | 113.81 x 131.22 x 106.08 | 0.173 x 0.200 x 0.162 |
| Slotted_Screwdriver | 5 | 148.75 x 22.82 x 38.34 | 0.200 x 0.031 x 0.052 |
| FizzyTabletsCalcium | 3 | 33.04 x 164.98 x 33.23 | 0.040 x 0.200 x 0.040 |
| MelforBottle | 7 | 74.85 x 269.39 x 74.50 | 0.056 x 0.200 x 0.055 |
| Deodorant | 5 | 46.60 x 100.44 x 46.77 | 0.093 x 0.200 x 0.093 |
| CokePlasticLarge | 5 | 80.81 x 322.05 x 82.04 | 0.050 x 0.200 x 0.051 |
| Moon | 16 | 217.90 x 183.74 x 124.64 | 0.200 x 0.169 x 0.114 |

The exact values and transforms are available in `model_dimensions_and_scaling.csv` and the scaled JSON files.

## 3. Grasp representation and generation protocol

Let a grasp be

```text
g = (T, R, theta)
```

where `T` and `R` are the global hand pose and `theta` contains the hand joint angles. The implementation should accept a URDF/MJCF hand model and provide forward kinematics for a fixed set of sampled hand surface points.

### Recommended contact candidates

Use a small fixed set of hand contact candidates:

- distal pad of each finger;
- middle phalanx pad of each finger;
- two or three palm points.

For each optimization run, choose four candidates from distinct fingers or from three fingers plus the palm. Four contacts follow the simplified setting used by DexGraspNet and keep optimization inexpensive.

### Target-superquadric selection

For each candidate grasp:

1. Select one grasp target superquadric or a connected target group.
2. Compute contact attraction only against the selected target.
3. Compute collision against the union of all reconstructed superquadrics.
4. Do not separately target near-duplicate superquadrics with high geometric overlap.

This distinction is important for Slotted_Screwdriver, where SQ1--SQ4 strongly overlap, and for Moon, which contains several overlapping local primitives.

### Initialization

Use the simple DexGraspNet initialization principle:

1. Start from a canonical open-hand pose.
2. Sample a point on an inflated AABB or convex hull around the target superquadric.
3. Orient the palm toward the target center.
4. Randomly rotate the hand around the approach direction.
5. Add small joint and pose perturbations.

Generate 20--40 initial poses per object. Optimize all candidates, filter invalid results, and select five diverse valid grasps per object.

## 4. Simplified optimization objective

All geometric distances should be normalized by the object reference length `L = 0.20 m`. This makes the energy dimensionless and allows one set of weights to be used for all eight objects.

The recommended objective is:

```text
E(g) = E_fc
     + 100 E_contact
     + 100 E_pen
     + 10 E_self
     + E_joint
```

The relative weights follow the design used in DexGraspNet. They should be treated as initial values and fixed across all eight objects after a small pilot test.

### 4.1 Contact attraction

For the four selected hand contact points `p_i(g)`:

```text
E_contact = (1/n) sum_i [d(p_i(g), S_target) / L]^2
```

where `d` is the distance to the target superquadric surface or target union. This term pulls the selected finger/palm points toward the intended grasp region.

### 4.2 Approximate force closure

Following DexGraspNet, construct the grasp matrix `G` from the current contact positions and use the target surface normals `c`:

```text
E_fc = ||G c||_2^2
```

This inexpensive differentiable term encourages the contact normals and moment arms to balance. A full friction-cone Q1 computation is not required during optimization and can instead be used for final evaluation.

### 4.3 Object penetration

Sample points `v` on the hand surface:

```text
E_pen = (1/|V|) sum_v ReLU[-sd(v, S_all) / L]^2
```

where `sd` is the signed distance to the union of all reconstructed superquadrics. This term penalizes hand-object penetration. For numerical robustness, the union may be converted once to a watertight triangle mesh and queried with a standard signed-distance implementation.

### 4.4 Hand self-penetration

For sampled points on non-adjacent hand links:

```text
E_self = mean ReLU[(delta_self - d_link) / L]^2
```

Use `delta_self = 2 mm` after scaling. Adjacent links that are expected to touch at joints must be excluded.

### 4.5 Joint limits and natural posture

```text
E_joint =
    sum_k ReLU(theta_k - theta_k_max)^2
  + sum_k ReLU(theta_k_min - theta_k)^2
  + 0.01 ||theta - theta_ref||_2^2
```

The first two terms enforce joint limits. The weak reference-pose term prevents twisted postures without dominating contact generation.

### Optimization algorithm

Use Adam for 1000--2000 iterations, followed by a short L-BFGS refinement if needed. This is simpler than MALA and follows DexGraspNet's observation that good initialization makes gradient-based optimization sufficient.

## 5. Selecting five grasps per object

After optimization, reject candidates that violate any hard condition:

- fewer than three contacting fingers;
- mean selected-contact distance greater than 3 mm;
- maximum object penetration greater than 2 mm;
- self-penetration greater than 2 mm;
- any joint outside its valid range.

Rank the remaining candidates using:

```text
score = Q1 - 0.5 penetration_m - 0.1 contact_distance_m
```

Then apply greedy diversity selection:

1. Select the highest-scoring grasp.
2. Reject candidates whose wrist pose and joint vector are too close to an already selected grasp.
3. Continue until five grasps are selected.

Suggested diversity thresholds:

- wrist translation difference: at least 15 mm, or
- wrist rotation difference: at least 15 degrees, or
- normalized joint-space distance: at least 0.10.

## 6. Quantitative evaluation metrics

The final experiment contains 8 objects x 5 grasps = 40 selected grasp poses.

### Primary metrics

1. **Simulation success rate (%)**
   Evaluate each grasp in MuJoCo under gravity in six axis-aligned directions. A grasp succeeds if the object remains held for a fixed duration without falling or losing all contacts. Report per-object and overall success rates.

2. **Force-closure quality (Q1)**
   Compute the Ferrari--Canny Q1 metric from contacts within a 2 mm threshold. Set Q1 to zero when penetration exceeds the allowed threshold.

3. **Maximum penetration depth (mm)**
   Report mean, standard deviation, and maximum over the 40 selected grasps.

4. **Mean contact distance (mm)**
   Mean distance between selected hand contact points and the target superquadric surface.

### Secondary metrics

5. **Valid generation rate (%)**
   Number of candidates passing all geometric filters divided by the total number of optimization initializations.

6. **Contact richness**
   Number of contacting fingers and number of distinct contacting links per grasp.

7. **Target consistency (%)**
   Fraction of intended contact points whose nearest reconstructed primitive is the selected target SQ or target group.

8. **Grasp diversity**
   Mean pairwise wrist translation, wrist rotation, and normalized joint-space distance among the five grasps for each object.

9. **Optimization time**
   Mean runtime per initialization and per accepted grasp.

### Minimum result table

For a concise paper presentation, the main table should report:

| Object | Valid rate | Simulation success | Mean Q1 | Max penetration | Contact distance |
|---|---:|---:|---:|---:|---:|

An aggregate row should report the mean over all eight objects. Diversity and runtime can be reported in supplementary material or a second compact table.

## 7. Recommended ablation

If computational time permits, include one small ablation:

- full objective;
- without `E_fc`;
- without target-superquadric assignment, using only the complete object union.

This directly tests whether the reconstructed part structure contributes to stable and region-specific grasp generation. Five grasps for two representative objects are sufficient for this ablation.

## 8. Implementation sequence

1. Import the hand URDF/MJCF and implement differentiable forward kinematics.
2. Sample and store hand contact candidates and collision points.
3. Convert each scaled multi-superquadric union to a collision mesh or implement a stable union signed-distance query.
4. Implement initialization around a selected target SQ.
5. Implement the five energy terms and optimize batches of initial poses.
6. Filter, rank, and diversity-select five grasps per object.
7. Validate all 40 grasps in MuJoCo and export metrics and visualizations.
