import json
import re
import os
import sys
import time
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import vertexai
from google.auth import default
from google.cloud import aiplatform_v1beta1, bigquery, storage
from vertexai.generative_models import GenerativeModel
import traceback
import subprocess


# ==============================================================================
# 1. PROMPT DEFINITIONS
# ==============================================================================

LOOKER_ANALYSIS_PROMPT = """
Analyze these Looker Studio dashboard SQL queries and extract comprehensive metrics information.

**IMPORTANT CONTEXT:**
All of the `SQL Samples` provided below originate from the **same Looker Studio dashboard**. These queries typically use the same underlying dataset, but may appear different due to dashboard controls like filters or parameters. Your primary goal is to identify the core, reusable metrics. Pay close attention to how different `WHERE` clauses or `CASE` statements might be creating variations of a single base metric. Look for these overlaps, as they are key consolidation opportunities.

**ERROR HANDLING INSTRUCTIONS:**
If you cannot fully and accurately generate the complete JSON structure as requested for any reason, your ENTIRE output should be ONLY the following single JSON object:
{{
  "dashboard_summary": {{
    "dashboard_id": "GENERATION_ERROR",
    "dashboard_name": "AI failed to process the request.",
    "business_domain": "other",
    "complexity_score": 0,
    "consolidation_score": 0,
    "score_reasoning": "The AI model could not generate a valid analysis for the provided SQL samples.",
    "total_metrics_identified": 0,
    "date_grain": "none"
  }},
  "dataset_analysis": null,
  "metrics": []
}}

**INPUT DATA:**
- Dashboard ID: {dashboard_id}
- Dashboard Name: {dashboard_name}
- SQL Samples: {sql_samples}

**OUTPUT REQUIREMENTS (JSON):**
Return a flat JSON structure with the following schema:
{{
  "dashboard_summary": {{
    "dashboard_id": "string",
    "dashboard_name": "string",
    "primary_data_sources": "project1.dataset1.table1;project2.dataset2.table2",
    "business_domain": "advertising|finance|consumer|operations|marketing|sales|product|hr|other",
    "complexity_score": "A score from 1-10 on the technical complexity of the dashboard's SQL.",
    "consolidation_score": "A score from 1-10 indicating how much this dashboard would benefit from consolidation.",
    "score_reasoning": "Briefly explain the reasoning behind the complexity and consolidation scores.",
    "total_metrics_identified": "number",
    "date_grain": "daily|weekly|monthly|quarterly|yearly|mixed|none"
  }},
  "dataset_analysis": {{
    "primary_analysis_sql": "A query that calculates several of the most important metrics together to provide a representative sample of the dashboard's primary function. It should use GROUP BY on key dimensions.",
    "structure_sql": "A query to understand the structure of the main source table. It should find the total record count, approximate unique counts of key IDs, and the min/max dates to understand the data's grain and timespan.",
    "validation_sql": "A simple query that performs quick syntactic and data freshness checks. For example, check if a key metric is > 0 or if the latest data is recent. It should return a 'PASS' or 'FAIL' status for each check. This query MUST always LIMIT 1.",
    "business_rules_sql": "A query that validates a core business rule found in the source SQL. For example, if there is a CASE statement that defines customer types, this query should return the counts for each type."
  }},
  "metrics": [
    {{
      "metric_id": "unique_identifier_for_this_specific_variation",
      "base_metric_id": "A common identifier for metrics that represent the same core concept (e.g., 'total_revenue'). This should be the same for 'YTD Revenue' and 'Rolling 12 Month Revenue'.",
      "metric_name": "Human Readable Name",
      "business_description": "what this metric represents in business terms",
      "gcp_project_name": "The GCP project ID of the primary table used for this metric.",
      "dataset_name": "The BigQuery dataset of the primary table.",
      "table_name": "The BigQuery table name of the primary table.",
      "sql_logic": "A complete and valid BigQuery query that calculates this single metric. It MUST include a full SELECT and FROM clause with the full `project.dataset.table` path. Example: 'SELECT COUNT(DISTINCT user_id) AS unique_users FROM `my-project.my_dataset.my_table` WHERE status = \\'active\\''",
      "metric_type": "dimension|measure|calculated_field|filter|aggregation|ratio|percentage",
      "is_kpi": "true|false",
      "business_criticality": "high|medium|low"
    }}
  ]
}}

**DATASET ANALYSIS REQUIREMENTS (BIGQUERY-COMPLIANT):**
**CRITICAL: Every SQL query generated in this section MUST end with a `LIMIT` clause. The limit must not exceed 12 rows (except for `validation_sql` which must `LIMIT 1`).**

**CRITICAL BIGQUERY SYNTAX ENFORCEMENT:**
- **NEVER** invent column names; only use names from the provided SQL.
- **ALWAYS** use `SAFE_CAST` before comparisons or functions to prevent type errors.
- **ENSURE** all non-aggregated columns in a SELECT are in the GROUP BY clause.
"""

def design_secondary_analysis_prompt():
    """
    Designs the secondary analysis prompt. This version is updated to accept
    a single, structured summary of all executed validation queries.
    """
    secondary_prompt = """
    SECONDARY LOOKER CONSOLIDATION ANALYSIS: SYNTHESIS AND INSIGHT
    ===============================================================

    You are an expert data architect. Your mission is to synthesize multiple sources of information to generate a strategic analysis of a Looker Studio dashboard.

    **GUIDING PRINCIPLES FOR ANALYSIS:**
    1.  **SYNTHESIZE AND COMPARE:** Your most important task is to find meaningful insights by comparing the inputs. Does the live data from the `sql_execution_summary` validate the assumptions in the `metrics_details`?
    2.  **INTERPRET, DON'T ASSUME:** An error or empty result could be due to permissions or a prior AI mistake. Frame your findings as "observations" and "areas for investigation."
    3.  **FOCUS ON PATTERNS:** Use the detailed metric inputs to identify anti-patterns like repeated hardcoded values or complex CASE statements that indicate a need for a lookup table.

    **INPUT DATA:**
    - Dashboard ID: {dashboard_id}
    - Dashboard Name: {dashboard_name}
    - Initial AI-Generated Dashboard Analysis: {original_dashboard_analysis}
    - Detailed Metric & Governance Data: {metrics_details}
    - SQL Query Execution Summary: {sql_execution_summary}

    **OUTPUT REQUIREMENTS (JSON):**
    {{
      "consolidation_analysis": {{
        "dashboard_id": "string",
        "dashboard_name": "string",
        "consolidation_priority": "high|medium|low",
        "key_synthesis": "A high-level summary of the most important findings discovered by comparing the initial analysis with the live data results. For example: 'The dashboard's primary revenue metric was validated, but the data reveals that the hardcoded CASE statement for 'Region' is missing several key regions, leading to understated results.'"
      }},
      "coding_practice_review": [
        {{
            "metric_id": "The ID of the metric with the issue.",
            "issue_type": "Hardcoded Logic|Anti-Pattern|Data Mismatch",
            "description": "A summary of the issue, synthesized with live data. Example: 'Metric uses a CASE statement to define 3 business channels, but live data shows 5 distinct channel values, indicating the logic is incomplete.'",
            "code_snippet": "The specific part of the sql_logic that demonstrates the issue.",
            "recommendation": "A suggested fix, e.g., 'Replace CASE statement with a join to a governed channel lookup table for maintainability.'"
        }}
      ],
      "investigation_points": [
          {{
              "point_of_interest": "e.g., 'primary_analysis_sql query failed' or 'Metric X has unexpected NULL values'",
              "possible_causes": ["e.g., 'SQL syntax error from initial prompt', 'Service account lacks permissions'"],
              "recommended_next_step": "e.g., 'Manually validate SQL syntax' or 'Check service account IAM permissions.'"
          }}
      ]
    }}
    """
    return secondary_prompt

# --- CORRECTED PROMPT FOR DASHBOARD-LEVEL CONSOLIDATION ---
DASHBOARD_METRIC_CONSOLIDATION_PROMPT = """
Act as an expert data architect. Your task is to analyze the following list of metrics, which ALL come from a single Looker Studio dashboard. Your goal is to create a refined, consolidated, and scalable set of "golden metrics" for this dashboard by focusing on their core SQL logic.

**CRITICAL INSTRUCTIONS:**
1.  **Analyze the 'metric_sql_core'**: This field contains the essential calculation. Use it to identify metrics with the same underlying business logic, even if their full `sql_logic` differs due to filters or date ranges.
2.  **Eliminate Redundancy**: Consolidate metrics that share an identical or semantically similar `metric_sql_core`.
3.  **Promote Scalability**: Where you find hardcoded `CASE` statements or long `IN` lists within the core logic, recommend creating a governed reference table and using a `JOIN` instead.
4.  **Standardize Logic**: For each group of similar metrics, create a single, authoritative "golden" version of the SQL.
5.  **Output ONLY JSON**: Your entire output must be a single JSON object.

**INPUT DATA:**
- Dashboard ID: {dashboard_id}
- Dashboard Name: {dashboard_name}
- Business Domain: {business_domain}
- A JSON list of all metric objects for this dashboard:
{all_metrics_json}

**OUTPUT REQUIREMENTS (JSON Schema):**
Return a single JSON object with one key: "dashboard_metric_strategy".

{{
  "dashboard_metric_strategy": {{
    "dashboard_id": "{dashboard_id}",
    "dashboard_name": "{dashboard_name}",
    "consolidated_metrics": [
      {{
        "metric_name": "Standardized Business Name (e.g., 'Total Revenue')",
        "business_description": "A clear, concise business definition of this metric.",
        "metric_type": "core_metric|specialized_metric",
        "sql_logic": "The complete, optimized, and valid BigQuery SQL for this metric. This should be the 'golden' version.",
        "original_metric_ids": ["metric_id_1", "metric_id_4", "metric_id_7"],
        "consolidation_reason": "A brief explanation of how this new metric consolidates the originals (e.g., 'Combined three variations of revenue, identified by their shared core logic, into a single, filterable metric.')."
      }}
    ],
    "recommendations": [
        {{
            "recommendation_type": "create_reference_table|other",
            "description": "A detailed recommendation. For a reference table, describe the business logic it would contain (e.g., 'Create a reference table 'dim_customer_channels' to replace the hardcoded CASE statement found in the core logic for identifying customer acquisition channels.').",
            "related_metric_names": ["Metric Name 1", "Metric Name 2"]
        }}
    ]
  }}
}}
"""

# ==============================================================================
# 2. DATA PREPARATION AND PARSING FUNCTIONS
# ==============================================================================

def prepare_looker_analysis_batch(df):
    """Convert dataframe to batch input format with structured Looker analysis prompt."""
    batch_data = []
    for dashboard_id, group in df.groupby('looker_studio_report_id'):
        dashboard_data = {
            "dashboard_id": dashboard_id,
            "dashboard_name": group.iloc[0]['looker_studio_report_name'],
            "sql_samples": []
        }
        for _, row in group.iterrows():
            dashboard_data["sql_samples"].append({
                "job_id": row['jobId'],
                "username": row['username'],
                "runtime_seconds": row['runtime_seconds'],
                "total_processed_bytes": row['totalProcessedBytes'] if pd.notna(row['totalProcessedBytes']) else None,
                "sql_query": row['query_text']
            })

        formatted_prompt = LOOKER_ANALYSIS_PROMPT.format(
            dashboard_id=dashboard_data["dashboard_id"],
            dashboard_name=dashboard_data["dashboard_name"],
            sql_samples=json.dumps(dashboard_data["sql_samples"], indent=2, default=str)
        )
        batch_data.append({"content": formatted_prompt})
    return batch_data


def convert_batch_results_to_dataset(results):
    """
    Parses batch prediction results based on the latest primary prompt schema.
    """
    print("🚀 Parsing batch results with updated schema...")

    dashboard_data, metrics_data, dataset_analysis_data = [], [], []

    for response_id, result in enumerate(results):
        try:
            raw_text = result.get('response', {}).get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            if not raw_text:
                continue

            json_text_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
            if not json_text_match:
                json_text_match = re.search(r'(\{.*?\})', raw_text, re.DOTALL)

            if not json_text_match:
                continue

            parsed_response = json.loads(json_text_match.group(1))

            summary = parsed_response.get('dashboard_summary')
            if not summary or summary.get('dashboard_id') == 'GENERATION_ERROR':
                continue

            dashboard_id = summary['dashboard_id']
            summary['response_id'] = response_id
            dashboard_data.append(summary)

            analysis = parsed_response.get('dataset_analysis')
            if analysis:
                analysis['response_id'] = response_id
                analysis['dashboard_id'] = dashboard_id
                dataset_analysis_data.append(analysis)

            metrics = parsed_response.get('metrics')
            if metrics:
                for metric in metrics:
                    metric.update({'response_id': response_id, 'dashboard_id': dashboard_id})
                    metrics_data.append(metric)

        except Exception as e:
            print(f"✗ CRITICAL FAILURE on response {response_id}: {e}")

    datasets = {
        'dashboards': pd.DataFrame(dashboard_data),
        'metrics': pd.DataFrame(metrics_data),
        'dataset_analysis': pd.DataFrame(dataset_analysis_data),
    }

    print("✅ Parsing complete.")
    return datasets


def create_unified_dataset(datasets):
    """
    Creates a unified dataset from the primary analysis outputs.
    """
    print("🔗 Creating unified dataset from analysis results...")

    dashboards_df = datasets.get('dashboards')
    metrics_df = datasets.get('metrics')

    if dashboards_df is None or dashboards_df.empty:
        print("⚠️ No dashboard data found to create a unified dataset.")
        return pd.DataFrame()

    unified_records = []
    for _, dashboard in dashboards_df.iterrows():
        dashboard_id = dashboard['dashboard_id']
        dashboard_metrics = metrics_df[metrics_df['dashboard_id'] == dashboard_id] if metrics_df is not None else pd.DataFrame()

        metrics_summary_parts = []
        if not dashboard_metrics.empty:
            for _, metric in dashboard_metrics.head(5).iterrows():
                metrics_summary_parts.append(f"- {metric.get('metric_name', 'Unnamed Metric')}: {metric.get('business_description', 'No description.')}")
        metrics_summary_text = "\\n".join(metrics_summary_parts)

        dashboard_record = {
            'record_id': f"{dashboard_id}_summary",
            'record_type': 'dashboard_summary',
            'dashboard_id': dashboard_id,
            'dashboard_name': dashboard.get('dashboard_name'),
            'business_domain': dashboard.get('business_domain'),
            'complexity_score': dashboard.get('complexity_score'),
            'consolidation_score': dashboard.get('consolidation_score'),
            'total_metrics': len(dashboard_metrics),
            'full_context': f"Dashboard '{dashboard.get('dashboard_name')}' is in the {dashboard.get('business_domain')} domain. "
                            f"It has a complexity score of {dashboard.get('complexity_score')} and a consolidation score of {dashboard.get('consolidation_score')}. "
                            f"It contains {len(dashboard_metrics)} metrics in total. Key metrics include:\\n{metrics_summary_text}"
        }
        unified_records.append(dashboard_record)

    unified_df = pd.DataFrame(unified_records)
    print(f"✅ Unified dataset created successfully with {len(unified_df)} dashboard summary records.")
    return unified_df

# ==============================================================================
# 3. UTILITY & WORKFLOW FUNCTIONS
# ==============================================================================

def save_datasets_to_csv(datasets, output_folder="./data/"):
    """Saves all DataFrames in the datasets dictionary to CSV files."""
    os.makedirs(output_folder, exist_ok=True)
    if not isinstance(datasets, dict) or not datasets:
        print("⚠️ No datasets provided to save.")
        return False
    print(f"\n💾 Saving {len(datasets)} dataset(s) to CSV files in '{output_folder}'...")
    try:
        for name, df in datasets.items():
            if df is not None and not df.empty:
                output_path = os.path.join(output_folder, f"looker_analysis_{name}.csv")
                df.to_csv(output_path, index=False)
                print(f"✓ Saved {name}: {output_path} ({len(df)} rows)")
            else:
                print(f"⚠️ Skipped saving '{name}': The dataset was empty.")
        return True
    except Exception as e:
        print(f"✗ Failed to save datasets: {e}")
        return False


def analyse_results_summary(datasets):
    """Provides a quick summary analysis of the structured datasets."""
    print("\n" + "="*60)
    print("📊 LOOKER ANALYSIS RESULTS SUMMARY")
    print("="*60)

    if 'dashboards' in datasets and not datasets['dashboards'].empty:
        dashboards_df = datasets['dashboards']
        print(f"\n📋 DASHBOARDS ANALYZED: {len(dashboards_df)}")
        if 'business_domain' in dashboards_df.columns:
            print("\nDomains:")
            print(dashboards_df['business_domain'].value_counts().to_string())
        if 'complexity_score' in dashboards_df.columns:
            print(f"\nComplexity Scores (1-10):")
            print(f"  Average: {dashboards_df['complexity_score'].mean():.1f}")

    if 'metrics' in datasets and not datasets['metrics'].empty:
        metrics_df = datasets['metrics']
        print(f"\n\n📈 METRICS IDENTIFIED: {len(metrics_df)}")
        if 'metric_type' in metrics_df.columns:
            print("\nMetric Types:")
            print(metrics_df['metric_type'].value_counts().to_string())
    print("\n" + "="*60)

def format_emails_for_sql(email_list):
    """Format email list for use in SQL IN clauses."""
    if not email_list:
        return "''"
    formatted_emails = "', '".join(email_list)
    return f"'{formatted_emails}'"


def run_sql_file(file_path, replacements=None, client=None, project=None):
    """Read SQL file, apply replacements, and execute in BigQuery."""
    if replacements is None:
        replacements = {}
    try:
        sql = read_and_replace_sql(file_path, replacements)
        return execute_bq_query(sql, client, project)
    except Exception as e:
        print(f"✗ Failed to run SQL file {file_path}: {str(e)}")
        return None


def read_and_replace_sql(file_path, replacements=None):
    """Read SQL file and replace specified text patterns."""
    if replacements is None:
        replacements = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sql = f.read()
        for old_text, new_text in replacements.items():
            sql = sql.replace(old_text, new_text)
        return sql
    except FileNotFoundError:
        raise FileNotFoundError(f"SQL file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading SQL file {file_path}: {str(e)}")


def execute_bq_query(sql, client=None, project=None, to_dataframe=True):
    """Execute SQL query in BigQuery."""
    if client is None:
        client = bigquery.Client(project=project)
    try:
        if to_dataframe:
            df = client.query(sql).to_dataframe()
            print(f"✓ Query executed successfully - returned {len(df)} rows")
            return df
        else:
            job = client.query(sql)
            job.result()
            print(f"✓ Query executed successfully")
            return job
    except Exception as e:
        print(f"✗ Query execution failed: {str(e)}")
        return None

def run_gemini_batch_fast_slick(
    requests,
    project,
    display_name,
    input_gcs_uri,
    output_gcs_uri,
    model_name="gemini-1.5-flash",
    location="us-central1",
    max_output_tokens=8192,
    temperature=0.20
):
    """Run complete Gemini batch prediction workflow."""
    print("📤 Preparing batch input...")
    success = prepare_batch_input_for_gemini(requests, input_gcs_uri, temperature, max_output_tokens)
    if not success:
        return None

    print("🚀 Creating batch prediction job...")
    job = create_vertex_batch_prediction_job(
        project=project,
        display_name=display_name,
        model_name=f"publishers/google/models/{model_name}",
        input_gcs_uri=input_gcs_uri,
        output_gcs_uri=output_gcs_uri,
        location=location
    )
    if not job:
        return None

    final_status = wait_for_batch_completion_slick(job.name, location)

    if final_status and final_status['state'] == 'JOB_STATE_SUCCEEDED':
        print("📊 Reading results...")
        results = read_batch_prediction_results_fixed(output_gcs_uri)
        return results
    else:
        print("❌ Batch job failed or timed out")
        return None


def prepare_batch_input_for_gemini(requests, output_gcs_path, temperature=0.20, max_output_tokens=8192):
    """Prepare batch input in correct format for Gemini models."""
    try:
        batch_requests = []
        for request in requests:
            batch_requests.append({
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": request["content"]}]}],
                    "generation_config": {"temperature": temperature, "max_output_tokens": max_output_tokens}
                }
            })
        client = storage.Client()
        bucket_name, blob_path = output_gcs_path.replace('gs://', '').split('/', 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        jsonl_content = "\n".join(json.dumps(req) for req in batch_requests)
        blob.upload_from_string(jsonl_content, content_type='application/jsonl')
        print(f"✓ Batch input uploaded to {output_gcs_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to prepare batch input: {str(e)}")
        return False


def create_vertex_batch_prediction_job(project, display_name, model_name,
                                      input_gcs_uri, output_gcs_uri,
                                      location="us-central1",
                                      instances_format="jsonl",
                                      predictions_format="jsonl"):
    """Create batch prediction job using Vertex AI v1beta1 API."""
    try:
        api_endpoint = f"{location}-aiplatform.googleapis.com"
        client_options = {"api_endpoint": api_endpoint}
        client = aiplatform_v1beta1.JobServiceClient(client_options=client_options)
        batch_prediction_job = {
            "display_name": display_name,
            "model": model_name,
            "input_config": {
                "instances_format": instances_format,
                "gcs_source": {"uris": [input_gcs_uri]},
            },
            "output_config": {
                "predictions_format": predictions_format,
                "gcs_destination": {"output_uri_prefix": output_gcs_uri},
            }
        }
        parent = f"projects/{project}/locations/{location}"
        response = client.create_batch_prediction_job(
            parent=parent, batch_prediction_job=batch_prediction_job
        )
        print(f"✓ Batch prediction job created: {response.name}")
        return response
    except Exception as e:
        print(f"✗ Batch prediction job creation failed: {str(e)}")
        return None


def monitor_batch_prediction_job_quiet(job_name, location="us-central1"):
    """Monitor batch prediction job status - minimal output"""
    try:
        api_endpoint = f"{location}-aiplatform.googleapis.com"
        client_options = {"api_endpoint": api_endpoint}
        client = aiplatform_v1beta1.JobServiceClient(client_options=client_options)
        job = client.get_batch_prediction_job(name=job_name)
        return {
            'state': job.state.name,
        }
    except Exception as e:
        print(f"✗ Failed to get job status: {str(e)}")
        return None

def wait_for_batch_completion_slick(job_name, location="us-central1", check_interval=10, max_wait=7200):
    """Wait for batch prediction job with slick progress display"""
    print(f"Monitoring batch job: {job_name.split('/')[-1]}")
    start_time = time.time()
    completion_states = ['JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED']

    while time.time() - start_time < max_wait:
        status = monitor_batch_prediction_job_quiet(job_name, location)
        if status:
            state = status['state']
            elapsed = int(time.time() - start_time)
            sys.stdout.write(f"\rStatus: {state} | Elapsed: {elapsed//60}m {elapsed%60}s")
            sys.stdout.flush()
            if state in completion_states:
                sys.stdout.write("\n")
                print(f"✓ Job completed with status: {state}")
                return status
        time.sleep(check_interval)
    sys.stdout.write("\n")
    print(f"✗ Job did not complete within {max_wait} seconds")
    return None

def read_batch_prediction_results_fixed(output_gcs_prefix):
    """Read batch prediction results from GCS."""
    try:
        client = storage.Client()
        bucket_name, prefix = output_gcs_prefix.replace('gs://', '').split('/', 1)
        bucket = client.bucket(bucket_name)
        blobs = [b for b in bucket.list_blobs(prefix=prefix) if b.name.endswith('.jsonl')]

        if not blobs:
            print("No prediction files found!")
            return []

        results = []
        for blob in blobs:
            content = blob.download_as_text()
            for line in content.strip().split('\n'):
                if line:
                    results.append(json.loads(line))
        print(f"✓ Successfully read {len(results)} predictions")
        return results
    except Exception as e:
        print(f"✗ Failed to read batch results: {e}")
        return []

def run_dashboard_level_consolidation(project, input_gcs_uri, output_gcs_uri, model_name):
    """
    Runs a more purposeful, dashboard-by-dashboard metric consolidation analysis,
    focusing on the 'metric_sql_core' for more accurate consolidation.
    """
    print("\n--- Starting Dashboard-Level Metric Consolidation Analysis (Core Logic) ---")
    try:
        metrics_df = pd.read_csv("./data/looker_analysis_metrics.csv")
        dashboards_df = pd.read_csv("./data/looker_analysis_dashboards.csv")

        metrics_with_context_df = pd.merge(
            metrics_df,
            dashboards_df[['dashboard_id', 'dashboard_name', 'business_domain']],
            on='dashboard_id',
            how='left'
        )
        print(f"✓ Loaded {len(metrics_df)} metrics across {metrics_with_context_df['dashboard_id'].nunique()} dashboards.")
    except FileNotFoundError as e:
        print(f"✗ ERROR: Could not find necessary CSV file: {e}. Please run the initial analysis first.")
        return None

    print(f"\n--- Preparing {metrics_with_context_df['dashboard_id'].nunique()} separate consolidation requests... ---")
    consolidation_requests = []
    for dashboard_id, group in metrics_with_context_df.groupby('dashboard_id'):
        dashboard_info = group.iloc[0]
        dashboard_name = dashboard_info['dashboard_name']
        business_domain = dashboard_info['business_domain']

        metrics_to_analyze = group[['metric_id', 'metric_name', 'business_description', 'metric_sql_core']].to_dict('records')
        all_metrics_json = json.dumps(metrics_to_analyze, indent=2)

        formatted_prompt = DASHBOARD_METRIC_CONSOLIDATION_PROMPT.format(
            dashboard_id=dashboard_id,
            dashboard_name=dashboard_name,
            business_domain=business_domain,
            all_metrics_json=all_metrics_json
        )
        consolidation_requests.append({'content': formatted_prompt})

    print("\n--- Executing batch analysis job with Gemini ---")
    consolidation_results = run_gemini_batch_fast_slick(
        requests=consolidation_requests,
        project=project,
        display_name="looker-dashboard-consolidation-core",
        input_gcs_uri=input_gcs_uri.replace('.jsonl', '_dashboard_consolidation.jsonl'),
        output_gcs_uri=output_gcs_uri.rstrip('/') + '_dashboard_consolidation/',
        model_name=model_name
    )
    if not consolidation_results:
        print("✗ Dashboard-level consolidation analysis failed.")
        return None

    print("\n--- Parsing results and saving to new CSV files ---")
    all_consolidated_metrics = []
    all_recommendations = []
    for result in consolidation_results:
        try:
            raw_text = result.get('response', {}).get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            json_text_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
            if not json_text_match:
                json_text_match = re.search(r'(\{.*?\})', raw_text, re.DOTALL)

            if json_text_match:
                parsed_response = json.loads(json_text_match.group(1))
                strategy = parsed_response.get('dashboard_metric_strategy', {})
                dashboard_id = strategy.get('dashboard_id')

                if 'consolidated_metrics' in strategy:
                    for metric in strategy['consolidated_metrics']:
                        metric['dashboard_id'] = dashboard_id
                        all_consolidated_metrics.append(metric)

                if 'recommendations' in strategy:
                    for rec in strategy['recommendations']:
                        rec['dashboard_id'] = dashboard_id
                        all_recommendations.append(rec)
        except Exception as e:
            print(f"⚠️ Warning: Failed to parse a response. Error: {e}")

    output_folder = "./data/secondary/"
    os.makedirs(output_folder, exist_ok=True)

    if all_consolidated_metrics:
        consolidated_df = pd.DataFrame(all_consolidated_metrics)
        output_path = os.path.join(output_folder, "dashboard_consolidated_metrics.csv")
        consolidated_df.to_csv(output_path, index=False)
        print(f"✓ Saved {len(consolidated_df)} consolidated metrics to: {output_path}")

    if all_recommendations:
        recommendations_df = pd.DataFrame(all_recommendations)
        output_path = os.path.join(output_folder, "dashboard_recommendations.csv")
        recommendations_df.to_csv(output_path, index=False)
        print(f"✓ Saved {len(recommendations_df)} recommendations to: {output_path}")

    return {
        "consolidated_metrics": pd.DataFrame(all_consolidated_metrics),
        "recommendations": pd.DataFrame(all_recommendations)
    }

# ==============================================================================
# 4. GIT UTILITY FUNCTIONS (FIX)
# ==============================================================================

def fix_github_authentication():
    """Fix GitHub authentication with Personal Access Token"""
    USERNAME = "richie-peters"
    TOKEN = input("Paste your GitHub Personal Access Token here: ")
    if not TOKEN or TOKEN.strip() == "":
        print("❌ No token provided!")
        return False
    TOKEN = TOKEN.strip()
    repo_url = f"https://{USERNAME}:{TOKEN}@github.com/richie-peters/looker-metrics.git"
    try:
        result = subprocess.run(
            f'git remote set-url origin {repo_url}',
            shell=True,
            capture_output=True,
            text=True,
            cwd='/content/looker-metrics'
        )
        if result.returncode == 0:
            print("✅ GitHub authentication updated successfully!")
            return True
        else:
            print(f"❌ Failed to set remote URL: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error setting up authentication: {e}")
        return False

def auto_git_push(commit_message="Auto-commit from Colab", files_to_add="."):
    """Automatically commit and push changes to git."""
    def run_git_command(command):
        return subprocess.run(command, shell=True, capture_output=True, text=True, cwd='/content/looker-metrics')

    print("🔄 Auto Git Push Starting...")
    status_result = run_git_command("git status --porcelain")
    if not status_result.stdout.strip():
        print("✅ No changes to commit.")
        return True

    run_git_command(f"git add {files_to_add}")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_result = run_git_command(f'git commit -m "{commit_message} - {timestamp}"')

    if "nothing to commit" in commit_result.stdout:
        print("✅ Nothing new to commit.")
        return True

    if commit_result.returncode != 0:
        print(f"❌ Failed to commit: {commit_result.stderr}")
        return False

    push_result = run_git_command("git push origin main")
    if push_result.returncode != 0:
        print(f"❌ Failed to push: {push_result.stderr}")
        return False

    print("✅ Successfully pushed to GitHub!")
    return True

def push_existing_commits():
    """Push any existing unpushed commits."""
    try:
        result = subprocess.run(
            'git push origin main',
            shell=True,
            capture_output=True,
            text=True,
            cwd='/content/looker-metrics'
        )
        if result.returncode == 0:
            print("✅ Successfully pushed existing commits!")
            return True
        else:
            if "Everything up-to-date" in result.stderr:
                print("✅ No unpushed commits found.")
                return True
            print(f"❌ Push failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return False