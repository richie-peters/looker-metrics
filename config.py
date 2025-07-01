"""
Configuration and Setup Module (EMAIL_LIST FIXED)
=================================================

This module handles environment setup, authentication, and project configurations.

Author: Data Team
Date: 2025-01-07
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

# GCS Paths
INPUT_GCS_URI = "gs://looker_metrics/input.jsonl"
OUTPUT_GCS_URI = "gs://looker_metrics/output/"

# Gemini Model Configuration
GEMINI_MODEL_NAME = "gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 65000
TEMPERATURE = 0.20

# SQL File Path
LOOKER_SQL_FILE = "./sql/looker_sql.txt"

# Email List for SQL Queries
EMAIL_LIST = [
    'santhosh.kanaparthi@news.com.au','desnica.kumar@news.com.au','romy.li@news.com.au','cecile.desphy@news.com.au','jeyaram.jawahar@news.com.au','nigel.aye@news.com.au','camille.shi@news.com.au','justin.guo@news.com.au','eric.loi@news.com.au','pravarthika.rathinakumar@news.com.au','kylie.lu@news.com.au','ritwik.deo@news.com.au','harry.mcauley@news.com.au'
]

# Directory Structure
BASE_PATH = "."
DIRECTORIES = {
    'functions': os.path.join(BASE_PATH, 'functions'),
    'scripts': os.path.join(BASE_PATH, 'scripts'),
    'data': os.path.join(BASE_PATH, 'data'),
    'sql': os.path.join(BASE_PATH, 'sql'),
    'results': os.path.join(BASE_PATH, 'results'),
    'logs': os.path.join(BASE_PATH, 'logs')
}

def setup_environment():
    """Configure global environment settings for analysis."""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
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
    validation_results = {
        'bigquery': False,
        'vertex_ai': False,
        'storage': False,
        'authentication': False,
        'projects_valid': False
    }

    try:
        bq_client = bigquery.Client(project=bq_project_id)
        bq_client.query("SELECT 1 as test").result()
        validation_results['bigquery'] = True
        print(f"✓ BigQuery access validated: {bq_project_id}")
    except Exception as e:
        print(f"✗ BigQuery validation failed: {str(e)}")

    try:
        vertexai.init(project=vertex_project_id, location=region)
        from vertexai.preview import generative_models
        model = generative_models.GenerativeModel("gemini-pro")
        validation_results['vertex_ai'] = True
        print(f"✓ Vertex AI access validated: {vertex_project_id}")
    except Exception as e:
        print(f"✗ Vertex AI validation failed: {str(e)}")

    try:
        storage_client = storage.Client()
        validation_results['storage'] = True
        print(f"✓ Cloud Storage access validated")
    except Exception as e:
        print(f"✗ Cloud Storage validation failed: {str(e)}")

    try:
        credentials, project = default()
        validation_results['authentication'] = True
        validation_results['default_project'] = project
        print(f"✓ Authentication validated: {project}")
    except Exception as e:
        print(f"✗ Authentication validation failed: {str(e)}")

    validation_results['projects_valid'] = all(validation_results[key] for key in ['bigquery', 'vertex_ai', 'storage', 'authentication'])

    if validation_results['projects_valid']:
        print(f"✓ All project validations passed")
    else:
        print(f"✗ Some validations failed - check configuration")

    return validation_results

def initialize_session():
    """Initialise a complete analysis session."""
    print("=" * 60)
    print("INITIALIZING LOOKER ANALYSIS SESSION")
    print("=" * 60)

    setup_environment()
    directories = create_directory_structure()
    validation_results = validate_project_setup()

    session_config = {
        'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'start_time': datetime.now(),
        'directories': directories,
        'project_config': {
            'bq_project': BQ_PROJECT_ID,
            'vertex_project': VERTEX_PROJECT_ID,
            'region': REGION
        },
        'validation_results': validation_results
    }

    print("\n" + "=" * 60)
    print("✓ ANALYSIS SESSION INITIALIZED")
    print("=" * 60)
    print(f"Session ID: {session_config['session_id']}")
    print(f"Start time: {session_config['start_time']}")
    print(f"BigQuery Project: {session_config['project_config']['bq_project']}")
    print(f"Vertex AI Project: {session_config['project_config']['vertex_project']}")

    return session_config

def print_session_summary(session_config):
    """Print a summary of the current session configuration."""
    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)

    print(f"BigQuery Project: {session_config['project_config']['bq_project']}")
    print(f"Vertex AI Project: {session_config['project_config']['vertex_project']}")
    print(f"Region: {session_config['project_config']['region']}")
    print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working Directory: {os.getcwd()}")

    print("\n" + "=" * 60)
