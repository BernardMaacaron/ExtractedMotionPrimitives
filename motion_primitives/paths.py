from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT.parent
REACHGRASP_ROOT = DATASETS_ROOT / 'ReachGrasp'
KIT_MOTIONS_ROOT = DATASETS_ROOT / 'KIT Motions'
ANDY_DATA_ROOT = DATASETS_ROOT / 'AndyData-lab-prescientTeleopICub'
ARM_GAZE_ROOT = DATASETS_ROOT / '3D-ARM-Gaze'
