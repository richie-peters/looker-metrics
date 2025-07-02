# Looker Studio Analysis Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-1e3a5f?style=for-the-badge&logo=python&logoColor=f5f5dc)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-BigQuery_|_Vertex_AI-1e3a5f?style=for-the-badge&logo=google-cloud&logoColor=f5f5dc)
![Colab](https://img.shields.io/badge/Google-Colab-1e3a5f?style=for-the-badge&logo=google-colab&logoColor=f5f5dc)

**Automated pipeline for analysing Looker Studio metrics using BigQuery and Gemini AI**

</div>

<br>

## Overview

Enterprise-grade pipeline that extracts data from BigQuery and processes it using Gemini AI for intelligent Looker Studio analytics.

<br>

## Features

- **BigQuery Integration** - Automated data extraction with SQL templates
- **AI Processing** - Gemini-powered analysis and insights  
- **Batch Operations** - Scalable processing for large datasets
- **CSV Export** - Structured output for further analysis

<br>

## Installation & Execution

### Step 1: Setup Repository
```python
# Clone and navigate to project
!git clone https://github.com/richie-peters/looker-metrics.git
import os
os.chdir('/content/looker-metrics')
Step 2: Install Dependencies# Install required packages
!pip install -q -r requirements.txt
Step 3: Authenticate# Google Cloud authentication
from google.colab import auth
auth.authenticate_user()
Step 4: ConfigureUpdate config.py with your settings:
BQ_PROJECT_ID - Your BigQuery project
VERTEX_PROJECT - Your Vertex AI project
INPUT_GCS_URI - Input bucket path
OUTPUT_GCS_URI - Output bucket path
Step 5: Run Pipeline# Execute the analysis
exec(open('main.py').read())
<br>ConfigurationParameterDescriptionExampleBQ_PROJECT_IDBigQuery project ID"my-project"VERTEX_PROJECTVertex AI project ID"my-vertex-project"INPUT_GCS_URIInput storage location"gs://bucket/input/"OUTPUT_GCS_URIOutput storage location"gs://bucket/output/"EMAIL_LISTTarget emails for analysisList of email strings<br>Project Structurelooker-metrics/
├── main.py              # Main execution script
├── config.py            # Configuration settings  
├── functions.py         # Core functions
├── sql/                 # SQL query templates
├── data/                # Output directory
└── requirements.txt     # Dependencies
<br>TroubleshootingAuthentication Issuesfrom google.colab import auth
auth.authenticate_user()
No Data Returned
Check BigQuery permissions
Verify email list configuration
Confirm dataset exists
Processing Errors
Enable Vertex AI API
Validate GCS bucket access
Check model availability
Debug Results# Check processing output
if 'global_results' in globals():
    print(f"Processed: {len(global_results)} records")
<br>Output
CSV Files - Generated in ./data/ directory
Console Logs - Execution summary and statistics
Global Variables - Results stored in global_results
<br>CustomisationEmail List: Edit EMAIL_LIST in main.pyEMAIL_LIST = ['user@company.com', 'user2@company.com']
Processing: Modify parameters in config.py and functions.py<br>Support
Review documentation and troubleshooting section
Check execution logs for error details
Contact Data Team for assistance
Submit GitHub issues for bugs
<br><div align="center">Enterprise Data Pipeline SolutionData Team | Production Ready</div>
```