# Looker Metrics Analysis

Automated analysis of Looker Studio dashboards using Gemini AI batch processing.

## Files

- `main.py` - Main execution pipeline
- `config.py` - Configuration and setup
- `functions.py` - Core analysis functions
- `sql/looker_sql.txt` - BigQuery SQL for data extraction

## Usage

1. Configure project IDs in `config.py`
2. Run `main.py` to execute the full pipeline
3. Results are saved to `./data/` folder

## Requirements

- Google Cloud Platform access (BigQuery, Vertex AI, Cloud Storage)
- Appropriate IAM permissions
- Python packages: pandas, google-cloud-bigquery, google-cloud-aiplatform, google-cloud-storage
