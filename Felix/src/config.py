"""Project configuration values."""
from pathlib import Path

# This file should live at: MedAI/Felix/src/config.py
# Then:
# - CURRENT_FILE = .../MedAI/Felix/src/config.py
# - SRC_DIR = .../MedAI/Felix/src
# - FELIX_ROOT = .../MedAI/Felix
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
FELIX_ROOT = SRC_DIR.parent

# Safety check: make sure this config is being run from Felix's folder
if FELIX_ROOT.name != "Felix":
    raise RuntimeError(
        f"Expected project root folder to be 'Felix', but got '{FELIX_ROOT.name}'. "
        f"Current resolved root is: {FELIX_ROOT}"
    )

PROJECT_ROOT = FELIX_ROOT

# Data paths — all scoped only to MedAI/Felix/
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output paths — all scoped only to MedAI/Felix/
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OOF_DIR = OUTPUTS_DIR / "oof"
TEST_PREDS_DIR = OUTPUTS_DIR / "test_preds"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FEATURES_DIR = OUTPUTS_DIR / "features"

# Default CV settings
DEFAULT_N_SPLITS = 5
DEFAULT_RANDOM_STATE = 42

# Optional known marker names for sanity checks
KNOWN_ATI_MARKERS = [
    "SPP1", "MRC1", "TNC", "HAVCR1", "WFDC2", "GDF15"
]


def ensure_directories() -> None:
    """Create standard project directories inside MedAI/Felix only."""
    dirs = [
        DATA_DIR,
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUTS_DIR,
        OOF_DIR,
        TEST_PREDS_DIR,
        REPORTS_DIR,
        FEATURES_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)