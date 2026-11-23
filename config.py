# ==============================================================================
# CONFIGURATION FILE: config.py
# Centralized settings, file paths, feature sets, and hyperparameters for ML.
# ==============================================================================

import os

# Base project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")

# Database paths (supports environment variable override via DROPOUT_DB_PATH)
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.environ.get("DROPOUT_DB_PATH", os.path.join(DB_DIR, "dropout_system.db"))
SCHEMA_PATH = os.path.join(DB_DIR, "schema.sql")

# Output directory paths
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUTS_DIR, "tables")
MODEL_PATH = os.path.join(OUTPUTS_DIR, "model.joblib")
METRICS_PATH = os.path.join(OUTPUTS_DIR, "metrics.json")

# Flask Secret Key
SECRET_KEY = "dropout_early_warning_system_secret_key_be_it"

# Reproducibility seed
RANDOM_STATE = 42

# Dataset Target Column
TARGET_COL = "Target"

# ------------------------------------------------------------------------------
# FEATURE SET DEFINITIONS
# ------------------------------------------------------------------------------

# 1. APP_FEATURES (16 features used by the website and the final model)
APP_FEATURES = [
    "Gender",
    "Age at enrollment",
    "Scholarship holder",
    "Tuition fees up to date",
    "Debtor",
    "Displaced",
    "Admission grade",
    "Previous qualification (grade)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)"
]

# 2. ENROLMENT_FEATURES (8 features available at student enrollment time)
ENROLMENT_FEATURES = [
    "Gender",
    "Age at enrollment",
    "Scholarship holder",
    "Tuition fees up to date",
    "Debtor",
    "Displaced",
    "Admission grade",
    "Previous qualification (grade)"
]

# 3. EARLY_FEATURES (12 features: 8 enrolment features + 4 1st-semester features)
EARLY_FEATURES = ENROLMENT_FEATURES + [
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)"
]

# ------------------------------------------------------------------------------
# RISK BAND THRESHOLDS
# Low < 0.33, Medium 0.33 to 0.66, High >= 0.66
# ------------------------------------------------------------------------------
RISK_LOW_THRESHOLD = 0.33
RISK_HIGH_THRESHOLD = 0.66
