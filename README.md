# Extracted Motion Primitives

SOC-independent dataset loading, preprocessing, and primitive extraction live here.
ARC-IIT owns networks, readouts, gains, training, and analyses that compare against those artifacts.
Raw datasets stay in their original directories.

## 3D-ARM-Gaze (DBAS22)

`motion_primitives.arm_gaze` loads the sibling `3D-ARM-Gaze/` release. Its default is the
authors' seven-angle **custom/displayed arm**, in radians. `arm='virtual'` selects the corrected
sensor-derived arm instead; these are different measurements. Raw data remain untouched.

```python
from motion_primitives import arm_gaze

subjects = arm_gaze.get_available_subjects()
trials = arm_gaze.load_arm_joint_data('s1', 'initial_acquisition')
time, joints, info = trials[0]  # elapsed seconds, seven named angles, source target ID and bounds
raw = arm_gaze.load_recording('s1', 'initial_acquisition')  # native target/head/gaze/arm channels
```

```bash
python scripts/prepare_arm_gaze.py
# Separate alternative measurement; never overwrite or silently mix collections:
python scripts/prepare_arm_gaze.py --arm virtual --output collections/3D-ARM-Gaze/virtual-phase200-v1
```

The default output is `collections/3D-ARM-Gaze/custom-phase200-v1/`. Like ReachGrasp it contains
`manifest.json`, `movements.csv`, `native.npz`, and `phase.npz` (`q`, endpoint baseline `h`, residual `f`).
It also contains every target's selection decision in `trial_audit.csv`, recording summaries in
`recordings.csv`, and native target/head/end-effector/gaze arrays in `context/<subject>/<phase>.npz`.
Context arrays use original sample indices; slice using the movement's source bounds. Invalid eye
measurements are NaN with validity masks, and sample-level tracking annotations are preserved.

Selection uses published target exclusions, removes pauses plus two samples and neutral-posture
intervals, and rejects interrupted/incomplete trials. Both repeated targets in the paired condition
and unsuccessful trials remain. These are **target-presentation trials, including target holding**,
not velocity-segmented reaches. Native timing is retained; phase interpolation is linear with no
smoothing or scaling. The seven joint coordinates are not interchangeable with ReachGrasp's.

Open `notebooks/00_3D_ARM_Gaze_Quality.ipynb` for selection, tracking, and gaze-quality checks.
`notebooks/02_3D_ARM_Gaze_Preprocessing.ipynb` follows ReachGrasp's Phase 1 analysis: native versus
phase timing, endpoint baseline and residual, and endpoint-distance associations with whole trajectories
and residuals. Both read the canonical collection independently; neither runs the other notebook.
Figures and tables go to `notebooks/results/<notebook-stem>/<collection-id>/`.
The preprocessing notebook defaults to all 153,799,491 trial pairs; its full distance arrays, ranking,
and plotting require substantial RAM. A labelled subset validation is not a full-cohort result.
See [3D_ARM_Gaze.md](3D_ARM_Gaze.md) for the measured inventory and preprocessing details.

## Layout

```text
motion_primitives/                  dataset adapters, phase/endpoint tools, existing FADA fitter
riemannian/                        existing mass-metric and geodesic extraction code
scripts/prepare_reachgrasp.py       build a canonical collection directly from recordings
scripts/*Riemannian*.py             existing Riemannian extraction entrypoints
scripts/evaluate_riemPrim_params.py existing parameter evaluation
notebooks/01_ReachGrasp_Preprocessing.ipynb
notebooks/00_3D_ARM_Gaze_Quality.ipynb
notebooks/02_3D_ARM_Gaze_Preprocessing.ipynb
notebooks/reach_grasp_geodesics.ipynb
notebooks/plot_riemPrim_nni_results.ipynb
nni/                               Riemannian parameter search
collections/<dataset-or-mixture>/<collection-id>/
    manifest.json                  sources, hashes, units, coordinates, preprocessing, exclusions
    movements.csv                  ordered dataset/subject/action/repetition identities
    native.npz                     native variable-length trajectories and timestamps
    phase.npz                      common-phase q, baseline h, residual f, endpoints
results/<dataset-or-mixture>/<method>/<run-id>/
    ...                            method-specific arrays, provenance, tables, figures
```

A collection describes the input cohort and preprocessing. A result run describes a method fitted or
measured on that collection. Method runs should record the ordered input collection paths, their
manifest hashes, selected joints, representation (`q` or `f`), whole-movement split, parameters,
software version/code hashes, and seed. Use a new run ID when those choices change.
For example, future FADA results belong in `results/ReachGrasp/fada-st/<run-id>/`; a mixed fit belongs
in `results/<explicit-mixture-name>/fada-st/<run-id>/`. Dataset names alone never define a cohort.

## Install and run

Use the existing scientific environment. No ARC-IIT import is needed here.

```bash
python -m pip install --no-deps --no-build-isolation -e ../ReachGrasp
python -m pip install --no-deps --no-build-isolation -e .
python scripts/prepare_reachgrasp.py --exclude sub-09/HC
python -m pytest -q tests
```

The default source is the sibling `ReachGrasp/` directory; `--dataset-path` overrides it. The default
output is `collections/ReachGrasp/vicon-phase200-v1`. Existing output directories are rejected to
prevent overwriting a collection. Use `--samples N --output collections/ReachGrasp/<new-id>` to
change the phase grid. The notebook reads saved collections and can run independently of preparation.

The exported cohort has **1,434 movements, 10 subjects, 16 action labels, and all 31 Vicon channels**.
`sub-09/HC` is explicitly excluded because its movement samples contain non-finite values. There is
no automatic exclusion, imputation, smoothing, amplitude scaling, or near-zero clamping in this
export. The command without `--exclude` fails on that recording. Exclusions are stored in the manifest.
The dataset's movement convention is the interval from one task's end to the next task's start,
using stop-exclusive cuts; it is not the task execution interval. Source sample bounds are in each row.

## Scientific steps: what, how, reference, software, output

### 1. Preserve measured movements

**What:** retain each complete movement in native time and physical units. **How:** call the existing
ReachGrasp loader with movement extraction enabled and clamping disabled; retain all 31 channels.
**Reference:** the dataset's `timeCuts.json`, channel TSV files, and loader's segmentation convention.
**Software:** `reachgrasp.data_loading`, pandas, NumPy. **Output:** `native.npz` and `movements.csv`.
The standardized joint names and original source channel names are both recorded in the manifest.

`native.npz` contains `q` with shape `(sum(T_a), D)`, `time` with shape `(sum(T_a),)`, and `offsets`
with shape `(A+1,)`. Movement `a` occupies `offsets[a]:offsets[a+1]`. Its elapsed timestamps start
at zero, its duration is `time[-1]-time[0]`, and source indices link back to recording time.
`movement_ids` and `joint_names` explicitly identify both axes. No object arrays or pickle are needed.

### 2. Construct a separate phase representation

**What:** compare trajectories on a common phase grid while preserving native duration separately.
**How:** set `phi=(t-t[0])/(t[-1]-t[0])`; linearly interpolate each channel to 200 evenly spaced
points including both endpoints. **Reference:** standard linear interpolation, no specialist model.
**Software:** `numpy.interp`. **Output:** `phase.npz:q` with shape `(movement, phase, joint)`.
Phase normalization removes duration from the comparison; it does not change the native recordings.
The older SOC interpolation helper still uses its original Savitzky–Golay/cubic procedure, now in
`motion_primitives.randg`; the two operations remain scientifically distinct.

### 3. Separate endpoints and observed shape

**What:** test whether endpoint context accounts for trajectory differences. **How:** compute
`h(phi) = q_start + phi * (q_goal - q_start)` and `f = q - h`.
**Reference:** the elementary/shape decomposition of Zhou, Gao & Asfour (2019),
[Learning Via-Point Movement Primitives](https://h2t.iar.kit.edu/pdf/Zhou2019.pdf), as specified in the
project plan. This is a direct observed residual, not a fitted VMP or discovered primitive.
**Software:** NumPy. **Output:** `phase.npz:h`, `f`, `q_start`, and `q_goal`; `q = h + f`.
A contextual target reconstructs `h + f`; the shape alone is not an absolute joint trajectory.

### 4. Diagnose endpoint dependence

**What:** compare endpoint separation with whole-trajectory and residual separation. **How:** use
`||q0_a-q0_b|| + ||qg_a-qg_b||` for endpoints and `sqrt(sum_t ||x_a-x_b||² / T)` for trajectories.
**Reference:** the project's proposed diagnostic. **Software:** SciPy `pdist`, Pearson/Spearman
coefficients, Matplotlib. **Output:** pair distances, correlation tables, provenance, and figures in
`results/ReachGrasp/endpoint_diagnostic/vicon-phase200-v1`. Pair distances are dependent; the notebook
reports descriptive coefficients, not significance tests. This step does not establish primitives.

## More datasets and mixtures

Every movement ID includes its dataset, e.g. `ReachGrasp/sub-01/Cyl/rep-00`. The table keeps dataset,
subject, action, repetition, duration, and source recording separately. Existing KIT and Andy loaders
were relocated here too; their new canonical exports have not been generated.

```python
from motion_primitives.preprocessing import load_collections

collection = load_collections([path_a, path_b], joint_names=common_ordered_joints)
Q, F = collection['q'], collection['f']
metadata = collection['metadata']  # includes dataset and source collection for every movement
```

Mixing requires explicit matching joint names, physical units, coordinate convention, phase sample
count, and preprocessing. The loader selects/reorders requested channels and rejects incompatible
collections and duplicate movement IDs. It does not assume that a human arm and an iCub arm share
coordinates because their tensors happen to have the same shape. A cross-body mapping is a separate
scientific operation. Preserve source dataset identity in splits; never split random time samples.

## Existing extraction methods and migration boundary

The FADA fitter is the existing Python implementation relocated from ARC-IIT, with its numerical
algorithm preserved. It is **not** the authors' official MATLAB FADA-T release; see
[Chiovetto et al. (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9510628/).
Using official FADA-T as proposed in the plan remains a separate fitting step. No new FADA or SCA
cohort sweep is claimed by this migration.

Riemannian extraction retains the original mass-metric/geodesic computation, smoothing, parameter
search, and native timing. It requires the existing `rbdl` environment. Its outputs default to
`results/ReachGrasp/riemannian/`; the migrated old results in that directory are historical outputs,
not fresh runs. The NNI bridge requires the existing `nni` environment. Run
`python scripts/extract_riemannian_primitives.py --help` for the single-recording entrypoint.
The full runner consumes explicit extraction parameters via `--params-file`.

The old FADA runner and 7c1/7c2 result notebooks in ARC-IIT remain SOC-cohort controls: their input
selection and scaling are owned by gain/readout manifests. They now import the external fitter.
Likewise, `7f_KinematicSCAControl` and `RiemannianAnalysis` remain in ARC-IIT because they include
SOC comparisons or gain geometry. Their old results are not silently relabelled as this unscaled
31-channel collection. New SOC-independent scientific work and results belong here.
