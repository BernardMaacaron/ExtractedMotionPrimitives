# 3D-ARM-Gaze / DBAS22: inventory and preprocessing

The source release is `/home/bmaacaron-iit.local/Documents/Datasets/3D-ARM-Gaze/`.
The inventory and results below were measured from the complete local release on 2026-09-17.
Raw files were read without modification.

## Dataset composition

- **20 right-handed participants**, `s1` through `s20`.
- **60 reaching recordings**: three conditions per participant, comprising **5,196,667 samples**
  and approximately **17.06 hours** including pauses and neutral-posture procedures.
- **20 additional range-of-motion recordings**, with per-subject seven-angle minimum/maximum CSVs.
- Participant metadata contain arm side, body height, segment dimensions, and calibration transforms.
  They do not supply a general demographic table with ages or sex.
- **18,075 planned targets** across the target-definition JSONs. The recordings contain
  **18,137 target presentations/IDs**, including initialization and terminal IDs outside those sets.
- **25.50 GB of extracted JSON**, including calibration recordings; ZIP archives are separate copies.
- Native acquisition is approximately **90 Hz**, with irregular timestamps. The median of retained
  trials' median-derived rates is **91.02 Hz**; rates vary and are not replaced by a nominal clock.

| Condition | Meaning | Planned targets | Retained trials | Validated trials | Median trial duration |
|---|---|---:|---:|---:|---:|
| `initial_acquisition` | Initial target acquisition | 6,115 | 5,809 | 5,460 | 2.296 s |
| `test_RNP_after_pauses` | Return to neutral posture after pauses | 4,000 | 3,913 | 3,839 | 1.922 s |
| `test_RNP_after_targets_pairs` | Return to neutral posture after target pairs | 7,960 | 7,817 | 7,708 | 1.911 s |
| Total | | 18,075 | **17,539** | **17,007** | |

Each participant has 299–322 planned initial targets, 200 pause-condition targets, and 398
paired-condition targets. The paired protocol repeats the target around the neutral-posture reset.
These condition names are experimental protocols, not ReachGrasp-style action labels.

The recordings include target position/orientation, target success and pause events; sensor,
virtual, and custom arm poses; head/trunk/opposite-shoulder information; left/right eye rays and
binocular focus. Additional neutral-posture cube/opposite-arm fields appear for participants s13–s20.
`load_recording(..., fields=...)` can read additional published fields explicitly.

## Coordinates and measurements

The default seven outputs describe the **custom arm displayed during teleoperation**:
shoulder pitch, shoulder roll, arm yaw, elbow pitch, forearm yaw, wrist pitch, wrist roll.
All are radians. The custom arm is distinct from the corrected sensor-derived virtual arm.
`arm='virtual'` extracts the latter using `shouVirtQuat`, never the documented inaccurate
`shouVirtQuatRaw`. There are no finger-joint trajectories comparable to ReachGrasp's hand channels.

For angle extraction, reflect Unity's x axis: an xyzw quaternion becomes `(-x, y, z, -w)`.
Following the supplied `rot_quat_utils.py:quats2config`, compute shoulder angles from
`R_root^-1 R_shoulder` in intrinsic XZY order; take elbow pitch from
`R_shoulder^-1 R_elbow`; then extract YXZ angles relative to `R_shoulder R_X(elbow_pitch)`.
This assigns the complete forearm yaw to the last three angles. The implementation matches the
authors' function on all **44,201 samples** of `s1/test_RNP_after_pauses`.

Saved context positions and quaternions retain the original **left-handed Unity coordinates**.
Positions are metres, quaternions are xyzw, and eye directions are dimensionless vectors in the head
frame. The world reference is arbitrary across participants. Joint-angle coordinates differ from
ReachGrasp Vicon coordinates; common anatomical labels alone do not make the datasets interchangeable.

## Trial selection and preprocessing

1. Segment by original `tgtNumber`, preserving subject, condition, target ID, and source sample bounds.
2. Apply the authors' target-level tracking exclusions. Remove pauses and their two following
   samples, the first two recording samples, neutral-posture intervals, and the final sample.
   Drop the final target as in the authors' analysis, and targets outside the target-definition set.
3. Require a contiguous active interval of at least two samples. Reject interrupted trials rather
   than joining data across a pause/reset. Preserve both repeated targets and unsuccessful attempts.
4. Recover seven joint angles and keep native timestamps, rebased to zero per trial. Reject
   non-finite angles/times or non-increasing time. Unwrap principal-angle branch crossings within
   each trial using `numpy.unwrap` before interpolation, preserving rotations modulo 2*pi.
   No smoothing, imputation, scaling, or clamping is performed. Primitive-discovery trials are trimmed to kinematic movement bounds using the configured hand-speed onset rule and target-arrival rule.
5. Separately map movement time to `phi=(t-t0)/(t1-t0)` and linearly interpolate to **200 endpoint-inclusive
   phase samples**. Compute `h(phi)=q_start+phi*(q_end-q_start)` and `f=q-h`.
6. Preserve native gaze/context data. Eye validity must equal 31; invalid eye origins/directions
   become NaN. Binocular focus requires both eyes valid. No interpolation crosses invalid gaze.

The resulting cohort contains **3,400,523 native movement samples**, approximately **11.13 hours**,
with trial durations **0.640–5.022 s**. These are whole active target presentations, including
target holding and validation, not isolated ballistic reaches. Published validation requires
the end effector to remain within 2 cm and 5 degrees for 0.6 s.

**Seven trials contain a principal-angle crossing near 2*pi.** These branches are unwrapped
before phase or training interpolation to avoid artificial full rotations. `angle_wrap_events`
records the corrections per trial; raw quaternion recordings remain unchanged.

The audit records 598 rejected presentations: **456 author-excluded targets**, **65 without two
active samples**, **15 final targets**, and **62 IDs outside the target sets**. No additional
interrupted-trial rejection occurred in this release. Reasons are exclusive and checked in that order.

The author sample-level freeze/jump annotations are preserved separately, rather than silently
turning them into whole-trial exclusions. **40,548 flagged samples remain in 8,120 retained trials**,
consistent with choosing the authors' target-level policy. A stricter tracking cohort must be an
explicit new selection. Maximum observed adjacent-sample time gap is approximately **0.201 s**.
The eye validity flags were reconstructed by the authors after acquisition, not directly recorded.

## Implementation and outputs

- `motion_primitives/arm_gaze.py`: streaming JSON reader, subject/condition discovery, trial audit,
  seven-angle conversion, and ReachGrasp-compatible `(time, joints, info)` loading. The arm loader
  caches up to 64 recordings for repeated training calls; returned trajectories are read-only by convention.
- `scripts/prepare_arm_gaze.py`: reproducible full-cohort export; refuses existing output directories.
- `collections/3D-ARM-Gaze/custom-phase200-v1/`: the completed **1.74 GB** collection, with
  `native.npz`, `phase.npz`, `movements.csv`, `trial_audit.csv`, `recordings.csv`, 60 context NPZs,
  and a manifest containing source/code SHA-256 hashes, units, coordinates, and selection policy.
- `notebooks/00_3D_ARM_Gaze_Quality.ipynb`: cohort selection, tracking annotations, gaze availability,
  and an example in native time. Existing quality outputs were moved under its matching results folder.
- `notebooks/02_3D_ARM_Gaze_Preprocessing.ipynb`: the same Phase 1 representation analysis as
  ReachGrasp: native/phase timing, `q=h+f`, and endpoint distances versus whole-trajectory and residual
  distances, with Pearson/Spearman coefficients and density plots. Both notebooks independently read
  the canonical collection. Outputs belong to `notebooks/results/<notebook-stem>/<collection-id>/`.
  Full pairwise analysis uses all 153,799,491 pairs; subset validation is labelled separately.
- ARC-IIT: registered source **`3D-ARM-Gaze`** in shared discovery/loading and the batched trainer.
  Readout calibration and training use the same linear time interpolation for this source. Other
  sources keep their existing interpolation. No SOC network/readout/gain was trained in this task.

```python
from motion_primitives import arm_gaze

trials = arm_gaze.load_arm_joint_data('s1', 'initial_acquisition')
time, joints, info = trials[0]
print(info['target_number'], joints.columns.tolist())
```

From the ExtractedMotionPrimitives directory, reproduce into a new collection:

```bash
python scripts/prepare_arm_gaze.py --output collections/3D-ARM-Gaze/custom-phase200-v1
```

For a separate virtual-arm collection, add `--arm virtual` and choose another output path.
No full virtual-arm collection was generated here.

## Verification

- Full 60-recording preprocessing completed successfully, beyond a synthetic smoke test.
- Every native trajectory was checked for finite values, monotonic time, sample counts, duration,
  and identity. All phase arrays satisfy `q=h+f` with zero endpoint residuals; all 60 context files
  have verified eye-validity masks. Real ARC-IIT loads were compared with the exported trajectories.
- Quality and Phase 1 native/phase/decomposition cells were executed in IPython with saved outputs.
  The pairwise Phase 1 calculation was validated on a labelled 2,000-trial subset; the full-cohort
  pairwise cells remain unexecuted because of available memory. This was direct cell execution
  rather than a separate Jupyter kernel run.
- Known-rotation tests verify handedness and all seven extracted coordinates.
- Synthetic trials verify exclusions, gaps, retained failures, original IDs, gaze masks, exported
  collection loading, and overwrite refusal.
- **14 focused checks passed**; **13 ExtractedMotionPrimitives tests passed**.
- ARC-IIT's full suite: **72 passed, 8 failed**. The failures reference old notebook/runner paths
  already moved before this task; they do not arise from the new dataset loading path.

## Supplied references

- `3D-ARM-Gaze/DBAS22_DocOnline/MainDataExplained.pdf`: variables, units, frames, validity, event semantics.
- `3D-ARM-Gaze/DBAS22_DocOnline/SummaryOfFiles.pdf`: source and exclusion-file roles.
- `DBAS22_CodeOnline/code_python/tool_box/rot_quat_utils.py:quats2config`: seven-angle convention.
- `DBAS22_CodeOnline/code_python/data_analysis/extraction_function.py`: reference selection policy.

The paired condition is intentionally kept in full here; the authors' comparison analysis removes
duplicate post-reset targets. This choice is explicit in the manifest, so this collection is not
claimed to reproduce the publication's statistical cohort.
