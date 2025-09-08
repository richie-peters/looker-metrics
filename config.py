"""
Configuration and Setup Module
=================================================
This module handles environment setup, authentication, and project configurations.
It now includes specific GCS paths for each stage of the analysis pipeline.
"""

import pandas as pd
import os
import sys
from datetime import datetime
import warnings
from google.cloud import bigquery, storage
import vertexai
from google.auth import default

# Project-specific configurations
BQ_PROJECT_ID = "ncau-data-newsquery-prd"
VERTEX_PROJECT_ID = "ncau-data-nprod-aitrain"
REGION = "us-central1"

# Gemini Model Configuration
GEMINI_MODEL_NAME = "gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 65000
TEMPERATURE = 0.10

# --- REVISED GCS PATHS FOR ISOLATED STAGES ---
# Each stage now has its own dedicated input and output GCS locations.
GCS_BUCKET = "gs://looker_metrics"
GCS_PATHS = {
    "stage_1_identification": {
        "input": os.path.join(GCS_BUCKET, "input/stage_1_identification.jsonl"),
        "output": os.path.join(GCS_BUCKET, "output/stage_1_identification/")
    },
    "stage_2_consolidation": {
        "input": os.path.join(GCS_BUCKET, "input/stage_2_consolidation.jsonl"),
        "output": os.path.join(GCS_BUCKET, "output/stage_2_consolidation/")
    },
    "stage_3_standardization": {
        "input": os.path.join(GCS_BUCKET, "input/stage_3_standardization.jsonl"),
        "output": os.path.join(GCS_BUCKET, "output/stage_3_standardization/")
    }
}
# -----------------------------------------------

# SQL File Path
LOOKER_SQL_FILE = "./sql/looker_sql.txt"

# Directory Structure
BASE_PATH = "."
DIRECTORIES = {
    'data': os.path.join(BASE_PATH, 'data'),
    'prompts': os.path.join(BASE_PATH, 'prompts'),
    'sql': os.path.join(BASE_PATH, 'sql'),
}

def setup_environment():
    """Configure global environment settings for analysis."""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    warnings.filterwarnings('ignore')
    print("✓ Environment configured successfully")

def create_directory_structure(base_path=BASE_PATH):
    """Create standard directory structure for the project."""
    for name, path in DIRECTORIES.items():
        os.makedirs(path, exist_ok=True)
        print(f"✓ Directory ready: {name} -> {path}")
    return DIRECTORIES

def validate_project_setup(bq_project_id=BQ_PROJECT_ID, vertex_project_id=VERTEX_PROJECT_ID, region=REGION):
    """Validate that GCP projects and services are properly configured."""
    print("\n--- Validating GCP Project Setup ---")
    try:
        credentials, project = default()
        print(f"✓ Authentication successful. Default project: {project}")
        bq_client = bigquery.Client(project=bq_project_id)
        bq_client.query("SELECT 1").result()
        print(f"✓ BigQuery access validated for project: {bq_project_id}")
        vertexai.init(project=vertex_project_id, location=region)
        print(f"✓ Vertex AI access validated for project: {vertex_project_id}")
        print("✓ All project validations passed.")
        return True
    except Exception as e:
        print(f"✗ GCP validation failed: {str(e)}")
        return False

def initialize_session():
    """Initialise a complete analysis session."""
    print("=" * 60)
    print("INITIALIZING LOOKER ANALYSIS SESSION")
    print("=" * 60)
    setup_environment()
    directories = create_directory_structure()
    is_valid = validate_project_setup()
    session_config = {
        'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'project_config': {'bq_project': BQ_PROJECT_ID, 'vertex_project': VERTEX_PROJECT_ID, 'region': REGION},
        'is_valid': is_valid
    }
    print("\n✓ ANALYSIS SESSION INITIALIZED")
    return session_config

def print_session_summary(session_config):
    """Print a summary of the current session configuration."""
    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print(f"  Session ID: {session_config['session_id']}")
    print(f"  BigQuery Project: {session_config['project_config']['bq_project']}")
    print(f"  Vertex AI Project: {session_config['project_config']['vertex_project']}")
    print("=" * 60)