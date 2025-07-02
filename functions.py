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
    Designs the secondary analysis prompt. This version is heavily updated to instruct
    the AI to synthesize and compare the initial analysis with the live data results,
    deriving deeper insights and value.
    """
    secondary_prompt = """
    SECONDARY LOOKER CONSOLIDATION ANALYSIS: SYNTHESIS AND INSIGHT
    ===============================================================

    You are an expert data architect. Your mission is to synthesize three distinct sources of information:
    1. An initial AI-generated analysis of a dashboard's structure and metrics.
    2. The detailed logic and known governance issues for each metric.
    3. The actual results from running validation queries against a live BigQuery database.

    **GUIDING PRINCIPLES FOR ANALYSIS:**
    1.  **SYNTHESIZE AND COMPARE:** Your most important task is to find meaningful insights by comparing the inputs.
        -   **Theory vs. Reality:** Does the data from `actual_sql_results` confirm the assumptions in `metrics_details`? If a metric's `sql_logic` expects a `user_id`, but the `sample_data` shows that column is 90% NULL, that is a critical finding to report.
        -   **Validate Governance Issues:** Use the `sample_data` to add context to the `governance_issues_found`. If an issue flagged a hardcoded list of countries, does the live data contain other countries that were missed by the hardcoding? Report this discrepancy.
        -   **Connect Errors to Causes:** If a validation query failed (e.g., 'table not found'), explicitly connect this back to the `metrics_details` that rely on that table and note the potential business impact.

    2.  **INTERPRET, DON'T ASSUME:** An error or empty result could be due to permissions, ephemeral tables, or a prior AI mistake. Frame your findings as observations for investigation.

    **INPUT DATA:**
    - Dashboard ID: {dashboard_id}
    - Dashboard Name: {dashboard_name}
    - Initial AI-Generated Dashboard Analysis: {original_dashboard_analysis}
    - Detailed Metric & Governance Data: {metrics_details}
    - Actual Data from SQL Execution: {actual_sql_results}

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
            "issue_type": "Hardcoded Value|Anti-Pattern|Data Mismatch",
            "description": "A summary of the issue, synthesized with live data. Example: 'Metric uses a CASE statement to define 3 business channels, but live data shows 5 distinct channel values, indicating the logic is incomplete.'",
            "code_snippet": "The specific part of the sql_logic that demonstrates the issue.",
            "recommendation": "A suggested fix, e.g., 'Replace CASE statement with a join to a governed channel lookup table to ensure all channels are captured.'"
        }}
      ],
      "investigation_points": [
          {{
              "point_of_interest": "e.g., 'primary_analysis_sql query failed' or 'Metric `revenue_by_channel` data appears incomplete'",
              "possible_causes": ["e.g., 'SQL syntax error from initial prompt', 'Service account lacks permissions', 'The CASE statement in the metric logic may be missing values found in the live data.'"],
              "recommended_next_step": "e.g., 'Manually validate SQL syntax' or 'Cross-reference CASE statement values with `SELECT DISTINCT` on the source column.'"
          }}
      ]
    }}
    """
    return secondary_prompt

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
        
        # Format the structured prompt for this dashboard
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
    Creates a separate, linkable dataframe for governance issues.
    """
    print("🚀 Parsing batch results with updated schema...")
    
    dashboard_data, metrics_data, dataset_analysis_data, governance_issues_data = [], [], [], []
    
    for response_id, result in enumerate(results):
        try:
            raw_text = result.get('response', {}).get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            if not raw_text:
                print(f"⚠️ Skipping response {response_id}: No raw text found.")
                continue

            json_text_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
            if not json_text_match:
                json_text_match = re.search(r'(\{.*?\})', raw_text, re.DOTALL)
            
            if not json_text_match:
                print(f"✗ Skipping response {response_id}: No valid JSON object found.")
                continue

            parsed_response = json.loads(json_text_match.group(1))
            
            summary = parsed_response.get('dashboard_summary')
            if not summary or summary.get('dashboard_id') == 'GENERATION_ERROR':
                print(f"⚠️ Response {response_id} reported a generation error from the AI.")
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
                    metric_id = metric.get('metric_id')
                    if 'governance_issues' in metric and metric['governance_issues']:
                        for issue in metric['governance_issues']:
                            issue.update({'response_id': response_id, 'dashboard_id': dashboard_id, 'metric_id': metric_id})
                            governance_issues_data.append(issue)
                    
                    metric.pop('governance_issues', None)
                    metric.update({'response_id': response_id, 'dashboard_id': dashboard_id})
                    metrics_data.append(metric)

        except Exception as e:
            print(f"✗ CRITICAL FAILURE on response {response_id}: {e}")

    datasets = {
        'dashboards': pd.DataFrame(dashboard_data),
        'metrics': pd.DataFrame(metrics_data),
        'dataset_analysis': pd.DataFrame(dataset_analysis_data),
        'governance_issues': pd.DataFrame(governance_issues_data)
    }
    
    print("✅ Parsing complete.")
    return datasets


def create_unified_dataset(datasets):
    """
    Creates a unified dataset from the primary analysis outputs for a knowledge base.
    """
    print("🔗 Creating unified dataset from analysis results...")

    dashboards_df = datasets.get('dashboards', pd.DataFrame())
    metrics_df = datasets.get('metrics', pd.DataFrame())
    governance_issues_df = datasets.get('governance_issues', pd.DataFrame())

    if dashboards_df.empty:
        print("⚠️ No dashboard data found to create a unified dataset.")
        return pd.DataFrame()

    # Pre-aggregate governance issues for easier lookup
    if governance_issues_df is not None and not governance_issues_df.empty:
        issue_summary = governance_issues_df.groupby('dashboard_id')['issue_type'].apply(
            lambda x: x.value_counts().to_dict()
        ).reset_index(name='issue_counts')
        dashboards_df = pd.merge(dashboards_df, issue_summary, on='dashboard_id', how='left')
    else:
        dashboards_df['issue_counts'] = None
    
    dashboards_df['issue_counts'] = dashboards_df['issue_counts'].fillna({})

    # Assemble records
    unified_records = []
    for _, dashboard in dashboards_df.iterrows():
        context_parts = [
            f"Dashboard Summary for '{dashboard.get('dashboard_name')}' (ID: {dashboard.get('dashboard_id')}).",
            f"Business Domain: {dashboard.get('business_domain')}.",
            f"Complexity Score: {dashboard.get('complexity_score')}/10. Consolidation Score: {dashboard.get('consolidation_score')}/10.",
            f"Reasoning: {dashboard.get('score_reasoning')}",
            f"Primary Data Sources: {dashboard.get('primary_data_sources')}.",
        ]

        # Add metrics summary to context
        dashboard_metrics = metrics_df[metrics_df['dashboard_id'] == dashboard.get('dashboard_id')] if metrics_df is not None else pd.DataFrame()
        if not dashboard_metrics.empty:
            context_parts.append(f"Contains {len(dashboard_metrics)} metrics.")
            for _, metric in dashboard_metrics.iterrows():
                context_parts.append(f"- Metric: '{metric.get('metric_name')}' ({metric.get('metric_id')}). Description: {metric.get('business_description')}. Executable SQL: {metric.get('sql_logic')}")
        
        dashboard_record = {
            'record_id': f"{dashboard.get('dashboard_id')}_summary",
            'record_type': 'dashboard',
            'dashboard_id': dashboard.get('dashboard_id'),
            'dashboard_name': dashboard.get('dashboard_name'),
            'full_context': " ".join(context_parts)
        }
        unified_records.append(dashboard_record)

    unified_df = pd.DataFrame(unified_records)
    print(f"✅ Unified dataset created successfully with {len(unified_df)} records.")
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

    if 'governance_issues' in datasets and not datasets['governance_issues'].empty:
        governance_df = datasets['governance_issues']
        print(f"\n\n🚨 GOVERNANCE ISSUES IDENTIFIED: {len(governance_df)}")
        if 'issue_type' in governance_df.columns:
            print("\nIssue Types:")
            print(governance_df['issue_type'].value_counts().to_string())

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
    max_output_tokens=65000,
    temperature=0.20
):
    """Run complete Gemini batch prediction workflow - slick version"""
    
    # Step 1: Prepare and upload input
    print("📤 Preparing batch input...")
    success = prepare_batch_input_for_gemini(requests, input_gcs_uri, temperature, max_output_tokens)
    if not success:
        return None
    
    # Step 2: Create batch job
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
    
    # Step 3: Wait for completion with slick progress
    final_status = wait_for_batch_completion_slick(job.name, location)
    
    if final_status and final_status['state'] == 'JOB_STATE_SUCCEEDED':
        print("📊 Reading results...")
        results = read_batch_prediction_results_fixed(output_gcs_uri)
        return results
    else:
        print("❌ Batch job failed or timed out")
        return None

def prepare_batch_input_for_gemini(requests, output_gcs_path, temperature=0.20, max_output_tokens=65000):
    """Prepare batch input in correct format for Gemini models."""
    try:
        batch_requests = []
        for i, request in enumerate(requests):
            batch_requests.append({
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": request["content"]}]
                        }
                    ],
                    "generation_config": {
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens
                    }
                }
            })
        client = storage.Client()
        bucket_name = output_gcs_path.replace('gs://', '').split('/')[0]
        blob_path = '/'.join(output_gcs_path.replace('gs://', '').split('/')[1:])
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        jsonl_content = ""
        for request in batch_requests:
            jsonl_content += json.dumps(request) + "\n"
        blob.upload_from_string(jsonl_content, content_type='application/json')
        print(f"✓ Batch input uploaded to {output_gcs_path}")
        print(f"  Records: {len(batch_requests)}")
        print(f"  Temperature: {temperature}, Max tokens: {max_output_tokens}")
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
        print(f"✓ Batch prediction job created: {display_name}")
        print(f"  Job name: {response.name}")
        print(f"  Model: {model_name}")
        print(f"  Input: {input_gcs_uri}")
        print(f"  Output: {output_gcs_uri}")
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
        
        if hasattr(job.state, 'name'):
            state_name = job.state.name
        else:
            state_name = str(job.state)
        
        return {
            'name': job.name,
            'display_name': job.display_name,
            'state': state_name,
            'create_time': job.create_time,
            'update_time': job.update_time,
            'error': job.error if hasattr(job, 'error') else None
        }
        
    except Exception as e:
        print(f"✗ Failed to get job status: {str(e)}")
        return None

def wait_for_batch_completion_slick(job_name, location="us-central1", check_interval=10, max_wait=7200):
    """Wait for batch prediction job with slick progress display"""
    print(f"Monitoring batch job: {job_name.split('/')[-1]}")
    start_time = time.time()
    
    completion_states = ['JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED']
    success_states = ['JOB_STATE_SUCCEEDED']
    
    while time.time() - start_time < max_wait:
        status = monitor_batch_prediction_job_quiet(job_name, location)
        
        if status:
            state = status['state']
            elapsed = int(time.time() - start_time)
            elapsed_str = f"{elapsed//60}m {elapsed%60}s"
            
            # Create status line
            if state == 'JOB_STATE_PENDING':
                status_line = f"⏳ PENDING | Elapsed: {elapsed_str}"
            elif state == 'JOB_STATE_RUNNING':
                status_line = f"🏃 RUNNING | Elapsed: {elapsed_str}"
            else:
                status_line = f"📊 {state} | Elapsed: {elapsed_str}"
            
            # Overwrite previous line
            sys.stdout.write(f"\r{status_line}")
            sys.stdout.flush()
            
            # Check if completed
            if state in completion_states:
                sys.stdout.write("\n")  # New line after completion
                if state in success_states:
                    print(f"✓ Job completed successfully!")
                else:
                    print(f"✗ Job completed with status: {state}")
                return status
        
        time.sleep(check_interval)
    
    sys.stdout.write("\n")
    print(f"✗ Job did not complete within {max_wait} seconds")
    return None

def read_batch_prediction_results_fixed(output_gcs_prefix):
    """Read batch prediction results from GCS - fixed version"""
    try:
        client = storage.Client()
        bucket_name = output_gcs_prefix.replace('gs://', '').split('/')[0]
        prefix = '/'.join(output_gcs_prefix.replace('gs://', '').split('/')[1:]).rstrip('/')
        
        print(f"Searching for results in gs://{bucket_name}/{prefix}")
        
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        # Find the main predictions.jsonl files (not incremental ones)
        prediction_files = [blob for blob in blobs 
                          if blob.name.endswith('predictions.jsonl') 
                          and 'incremental' not in blob.name 
                          and blob.size > 1000]
        
        if not prediction_files:
            print("No main predictions.jsonl files found!")
            return []
        
        # Sort by creation time and get the most recent
        prediction_files.sort(key=lambda x: x.time_created, reverse=True)
        
        results = []
        blob = prediction_files[0]
        print(f"Reading: {blob.name}")
        content = blob.download_as_text()
        
        for line_num, line in enumerate(content.strip().split('\n')):
            if line.strip():
                try:
                    result = json.loads(line)
                    results.append(result)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
        
        print(f"✓ Successfully read {len(results)} predictions")
        return results
        
    except Exception as e:
        print(f"✗ Failed to read batch results: {e}")
        return []

def run_gemini_batch_fast_slick(
    requests,
    project,
    display_name,
    input_gcs_uri,
    output_gcs_uri,
    model_name="gemini-1.5-flash",
    location="us-central1",
    max_output_tokens=65000,
    temperature=0.20
):
    """Run complete Gemini batch prediction workflow - slick version"""
    
    # Step 1: Prepare and upload input
    print("📤 Preparing batch input...")
    success = prepare_batch_input_for_gemini(requests, input_gcs_uri, temperature, max_output_tokens)
    if not success:
        return None
    
    # Step 2: Create batch job
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
    
    # Step 3: Wait for completion with slick progress
    final_status = wait_for_batch_completion_slick(job.name, location)
    
    if final_status and final_status['state'] == 'JOB_STATE_SUCCEEDED':
        print("📊 Reading results...")
        results = read_batch_prediction_results_fixed(output_gcs_uri)
        return results
    else:
        print("❌ Batch job failed or timed out")
        return None

def run_complete_analysis(dataset_df, project=None, max_dashboards=5):
    """Run complete analysis with all queries"""
    from google.cloud import bigquery
    
    # Create BigQuery client
    client = bigquery.Client(project=project) if project else bigquery.Client()
    
    print(f"Starting complete analysis for {min(max_dashboards or len(dataset_df), len(dataset_df))} dashboards...")
    
    # Execute all queries
    execution_results = execute_dataset_analysis_queries(
        dataset_df, 
        client=client, 
        project=project, 
        max_queries=max_dashboards
    )
    
    # Analyze performance
    performance_analysis = analyze_query_performance(execution_results['metadata'])
    
    # Combine results by type
    combined_results = combine_query_results_by_type(execution_results)
    
    return {
        'execution_results': execution_results,
        'combined_results': combined_results,
        'performance_analysis': performance_analysis
    }

def execute_dataset_analysis_queries(dataset_df, client=None, project=None, max_queries=None):
    """Execute all SQL queries from the dataset analysis DataFrame"""
    import pandas as pd
    
    if client is None:
        from google.cloud import bigquery
        client = bigquery.Client(project=project)
    
    # Query columns to process
    query_columns = ['primary_analysis_sql', 'structure_sql', 'validation_sql', 'business_rules_sql']
    
    all_results = []
    all_metadata = []
    
    # Limit queries if specified
    rows_to_process = dataset_df.head(max_queries) if max_queries else dataset_df
    
    print(f"Executing queries for {len(rows_to_process)} dashboards...")
    
    for idx, row in rows_to_process.iterrows():
        dashboard_id = row['dashboard_id']
        response_id = row['response_id']
        
        print(f"\n--- Processing Dashboard {idx + 1}/{len(rows_to_process)}: {dashboard_id[:8]}... ---")
        
        for query_type in query_columns:
            sql = row[query_type]
            
            if pd.isna(sql) or sql.strip() == '':
                print(f"⚠️ Skipping {query_type}: empty SQL")
                continue
            
            # Execute the query
            result = execute_sql_with_metadata(
                sql=sql,
                query_type=query_type,
                dashboard_id=dashboard_id,
                response_id=response_id,
                client=client,
                project=project
            )
            
            # Store results
            if result['data'] is not None:
                # Add linking columns to the data
                result['data']['dashboard_id'] = dashboard_id
                result['data']['response_id'] = response_id
                result['data']['query_type'] = query_type
                
                all_results.append({
                    'dashboard_id': dashboard_id,
                    'response_id': response_id,
                    'query_type': query_type,
                    'data': result['data']
                })
            
            all_metadata.append(result['metadata'])
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame(all_metadata)
    
    print(f"\n=== EXECUTION SUMMARY ===")
    print(f"Total queries attempted: {len(all_metadata)}")
    print(f"Successful queries: {len([m for m in all_metadata if m['execution_status'] == 'success'])}")
    print(f"Failed queries: {len([m for m in all_metadata if m['execution_status'] == 'failed'])}")
    
    if len(all_metadata) > 0:
        avg_time = sum([m['execution_time_seconds'] for m in all_metadata]) / len(all_metadata)
        total_rows = sum([m['row_count'] for m in all_metadata])
        print(f"Average execution time: {avg_time:.2f}s")
        print(f"Total rows returned: {total_rows}")
    
    return {
        'results': all_results,
        'metadata': metadata_df,
        'summary': {
            'total_queries': len(all_metadata),
            'successful_queries': len([m for m in all_metadata if m['execution_status'] == 'success']),
            'failed_queries': len([m for m in all_metadata if m['execution_status'] == 'failed']),
            'total_rows_returned': sum([m['row_count'] for m in all_metadata])
        }
    }

def execute_sql_with_metadata(sql, query_type, dashboard_id, response_id, client=None, project=None):
    """Execute a single SQL query and return results with metadata"""
    import time
    from datetime import datetime
    
    if client is None:
        from google.cloud import bigquery
        client = bigquery.Client(project=project)
    
    start_time = time.time()
    
    try:
        # Execute the query
        job = client.query(sql)
        results = job.result()
        
        # Convert to DataFrame
        df = results.to_dataframe()
        
        execution_time = time.time() - start_time
        
        # Create metadata
        metadata = {
            'dashboard_id': dashboard_id,
            'response_id': response_id,
            'query_type': query_type,
            'execution_status': 'success',
            'execution_time_seconds': round(execution_time, 2),
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'executed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'bytes_processed': job.total_bytes_processed if hasattr(job, 'total_bytes_processed') else None,
            'slot_ms': job.slot_millis if hasattr(job, 'slot_millis') else None,
            'error_message': None
        }
        
        print(f"✓ {query_type} for {dashboard_id}: {len(df)} rows, {execution_time:.2f}s")
        
        return {
            'data': df,
            'metadata': metadata,
            'sql': sql
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        
        metadata = {
            'dashboard_id': dashboard_id,
            'response_id': response_id,
            'query_type': query_type,
            'execution_status': 'failed',
            'execution_time_seconds': round(execution_time, 2),
            'row_count': 0,
            'column_count': 0,
            'columns': [],
            'executed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'bytes_processed': None,
            'slot_ms': None,
            'error_message': str(e)
        }
        
        print(f"✗ {query_type} for {dashboard_id}: FAILED - {str(e)[:100]}...")
        
        return {
            'data': None,
            'metadata': metadata,
            'sql': sql
        }


def analyze_query_performance(metadata_df):
    """Analyze the performance of executed queries"""
    import pandas as pd
    
    if metadata_df.empty:
        print("No metadata to analyze")
        return None
    
    print("\n=== QUERY PERFORMANCE ANALYSIS ===")
    
    # Success rates by query type
    success_by_type = metadata_df.groupby('query_type')['execution_status'].apply(
        lambda x: (x == 'success').sum() / len(x) * 100
    ).round(2)
    
    print("\nSuccess rates by query type:")
    for query_type, success_rate in success_by_type.items():
        print(f"  {query_type}: {success_rate}%")
    
    # Execution times by query type (successful only)
    successful_queries = metadata_df[metadata_df['execution_status'] == 'success']
    
    if not successful_queries.empty:
        print("\nExecution times (successful queries only):")
        time_stats = successful_queries.groupby('query_type')['execution_time_seconds'].agg(['mean', 'min', 'max']).round(2)
        for query_type, stats in time_stats.iterrows():
            print(f"  {query_type}: avg={stats['mean']}s, min={stats['min']}s, max={stats['max']}s")
        
        print("\nRow counts by query type:")
        row_stats = successful_queries.groupby('query_type')['row_count'].agg(['mean', 'min', 'max']).round(0)
        for query_type, stats in row_stats.iterrows():
            print(f"  {query_type}: avg={stats['mean']} rows, min={stats['min']}, max={stats['max']}")
    
    # Failed queries details
    failed_queries = metadata_df[metadata_df['execution_status'] == 'failed']
    if not failed_queries.empty:
        print(f"\n=== FAILED QUERIES ({len(failed_queries)}) ===")
        for _, row in failed_queries.iterrows():
            print(f"  {row['query_type']} [{row['dashboard_id'][:8]}...]: {row['error_message'][:100]}...")
    
    return {
        'success_rates': success_by_type,
        'performance_stats': time_stats if not successful_queries.empty else None,
        'failed_queries': failed_queries
    }

def prepare_secondary_batch_input_robust(unified_dataset, datasets, df_bq_results):
    """
    Prepares the batch input for the secondary prompt. This version is updated to
    gather and inject detailed metric logic and governance issues from the primary
    analysis, enabling a code practice review.
    """
    print("📊 PREPARING SECONDARY BATCH INPUT (ROBUST VERSION)")
    print("=" * 55)

    # Helper function remains the same
    def clean_for_json(obj):
        if isinstance(obj, Decimal): return float(obj)
        if isinstance(obj, (date, datetime)): return obj.isoformat()
        if pd.isna(obj) or obj is None: return None
        if isinstance(obj, np.generic): return obj.item()
        if isinstance(obj, dict): return {str(k): clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list): return [clean_for_json(item) for item in obj]
        return obj

    # Extract source dataframes
    dashboard_summaries = unified_dataset[unified_dataset['record_type'] == 'dashboard_summary']
    metrics_df = datasets.get('metrics', pd.DataFrame())
    governance_issues_df = datasets.get('governance_issues', pd.DataFrame())
    
    print(f"Processing {len(dashboard_summaries)} dashboard summaries")

    secondary_batch_data = []
    combined_results = df_bq_results.get('combined_results', {})

    for _, dashboard in dashboard_summaries.iterrows():
        dashboard_id = dashboard['dashboard_id']
        dashboard_name = dashboard.get('dashboard_name', 'Unknown Dashboard')
        print(f"  Processing: {dashboard_name[:50]}...")

        try:
            # 1. Prepare original dashboard analysis summary
            original_analysis = {k: v for k, v in dashboard.to_dict().items() if pd.notna(v)}

            # 2. NEW: Prepare detailed metrics and governance data
            metrics_details = []
            if not metrics_df.empty:
                dashboard_metrics = metrics_df[metrics_df['dashboard_id'] == dashboard_id]
                for _, metric in dashboard_metrics.iterrows():
                    metric_detail = metric.to_dict()
                    # Find and attach related governance issues
                    if not governance_issues_df.empty:
                        issues = governance_issues_df[governance_issues_df['metric_id'] == metric.get('metric_id')].to_dict('records')
                        metric_detail['governance_issues_found'] = issues
                    metrics_details.append(metric_detail)

            # 3. Prepare BQ execution results
            actual_results = {}
            for query_type in ['primary_analysis_sql', 'structure_sql', 'validation_sql', 'business_rules_sql']:
                if query_type in combined_results and combined_results.get(query_type) is not None:
                    query_result_df = combined_results[query_type]
                    dashboard_data = query_result_df[query_result_df['dashboard_id'] == dashboard_id]
                    if not dashboard_data.empty:
                        actual_results[query_type] = {'status': 'success', 'row_count': len(dashboard_data), 'sample_data': dashboard_data.head(2).to_dict('records')}
                    else:
                        actual_results[query_type] = {'status': 'no_data'}
                else:
                    actual_results[query_type] = {'status': 'not_available'}

            # 4. Format the prompt
            secondary_prompt = design_secondary_analysis_prompt()
            
            formatted_prompt = secondary_prompt.format(
                dashboard_id=dashboard_id,
                dashboard_name=dashboard_name,
                original_dashboard_analysis=json.dumps(clean_for_json(original_analysis)),
                metrics_details=json.dumps(clean_for_json(metrics_details)), # New input
                actual_sql_results=json.dumps(clean_for_json(actual_results))
            )

            secondary_batch_data.append({'content': formatted_prompt})
            print(f"    ✅ Successfully prepared request")

        except Exception as e:
            print(f"    ⚠️ Error preparing {dashboard_id}: {e}")
            continue

    print(f"\n✅ Successfully prepared {len(secondary_batch_data)} secondary analysis requests")
    return secondary_batch_data


def combine_query_results_by_type(execution_results):
    """Combine all results by query type for easier analysis"""
    import pandas as pd
    
    results_by_type = {}
    
    for result in execution_results['results']:
        query_type = result['query_type']
        
        if query_type not in results_by_type:
            results_by_type[query_type] = []
        
        results_by_type[query_type].append(result['data'])
    
    # Combine DataFrames for each query type
    combined_results = {}
    for query_type, dfs in results_by_type.items():
        if dfs:
            try:
                combined_df = pd.concat(dfs, ignore_index=True)
                combined_results[query_type] = combined_df
                print(f"✓ Combined {query_type}: {len(combined_df)} total rows from {len(dfs)} dashboards")
            except Exception as e:
                print(f"✗ Failed to combine {query_type}: {e}")
                combined_results[query_type] = None
    
    return combined_results

def analyze_query_performance(metadata_df):
    """Analyze the performance of executed queries"""
    import pandas as pd
    
    if metadata_df.empty:
        print("No metadata to analyze")
        return None
    
    print("\n=== QUERY PERFORMANCE ANALYSIS ===")
    
    # Success rates by query type
    success_by_type = metadata_df.groupby('query_type')['execution_status'].apply(
        lambda x: (x == 'success').sum() / len(x) * 100
    ).round(2)
    
    print("\nSuccess rates by query type:")
    for query_type, success_rate in success_by_type.items():
        print(f"  {query_type}: {success_rate}%")
    
    # Execution times by query type (successful only)
    successful_queries = metadata_df[metadata_df['execution_status'] == 'success']
    
    if not successful_queries.empty:
        print("\nExecution times (successful queries only):")
        time_stats = successful_queries.groupby('query_type')['execution_time_seconds'].agg(['mean', 'min', 'max']).round(2)
        for query_type, stats in time_stats.iterrows():
            print(f"  {query_type}: avg={stats['mean']}s, min={stats['min']}s, max={stats['max']}s")
        
        print("\nRow counts by query type:")
        row_stats = successful_queries.groupby('query_type')['row_count'].agg(['mean', 'min', 'max']).round(0)
        for query_type, stats in row_stats.iterrows():
            print(f"  {query_type}: avg={stats['mean']} rows, min={stats['min']}, max={stats['max']}")
    
    # Failed queries details
    failed_queries = metadata_df[metadata_df['execution_status'] == 'failed']
    if not failed_queries.empty:
        print(f"\n=== FAILED QUERIES ({len(failed_queries)}) ===")
        for _, row in failed_queries.iterrows():
            print(f"  {row['query_type']} [{row['dashboard_id'][:8]}...]: {row['error_message'][:100]}...")
    
    return {
        'success_rates': success_by_type,
        'performance_stats': time_stats if not successful_queries.empty else None,
        'failed_queries': failed_queries
    }


def auto_git_push(commit_message=None, files_to_add=".", include_timestamp=True, 
                  dry_run=False, force_push=False):
    """
    Automatically commit and push changes to git
    
    Args:
        commit_message (str): Custom commit message
        files_to_add (str): Files to add ('.' for all, or specific files)
        include_timestamp (bool): Add timestamp to commit message
        dry_run (bool): Show what would be done without doing it
        force_push (bool): Force push even if no changes detected
    """
    import subprocess
    import os
    from datetime import datetime
    
    def run_git_command(command, capture_output=True):
        """Run git command and return result"""
        try:
            if dry_run:
                print(f"[DRY RUN] Would run: {command}")
                return True, ""
            
            result = subprocess.run(command, shell=True, capture_output=capture_output, 
                                  text=True, cwd='/content/looker-metrics')
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)
    
    print("🔄 Auto Git Push Starting...")
    
    # Check if we're in a git repository
    success, output = run_git_command("git status --porcelain")
    if not success:
        print("❌ Not in a git repository or git error occurred")
        return False
    
    # Check if there are changes
    if not output.strip() and not force_push:
        print("✅ No changes to commit")
        return True
    
    print(f"📁 Changes detected:")
    if not dry_run:
        run_git_command("git status --short", capture_output=False)
    
    # Add files
    print(f"📤 Adding files: {files_to_add}")
    success, output = run_git_command(f"git add {files_to_add}")
    if not success:
        print(f"❌ Failed to add files: {output}")
        return False
    
    # Create commit message
    if commit_message is None:
        commit_message = "Auto-commit: Updated looker analysis code"
    
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_message = f"{commit_message} - {timestamp}"
    
    # Commit changes
    print(f"💾 Committing: {commit_message}")
    success, output = run_git_command(f'git commit -m "{commit_message}"')
    if not success:
        if "nothing to commit" in output:
            print("✅ Nothing new to commit")
            return True
        else:
            print(f"❌ Failed to commit: {output}")
            return False
    
    # Push to remote
    print("🚀 Pushing to GitHub...")
    success, output = run_git_command("git push origin main")
    if not success:
        print(f"❌ Failed to push: {output}")
        return False
    
    print("✅ Successfully pushed to GitHub!")
    return True

def setup_auto_git_config(username, email, token):
    """
    One-time setup for git configuration and authentication
    
    Args:
        username (str): Your GitHub username
        email (str): Your email address  
        token (str): Your GitHub Personal Access Token
    """
    import subprocess
    
    def run_command(command):
        try:
            result = subprocess.run(command, shell=True, capture_output=True, 
                                  text=True, cwd='/content/looker-metrics')
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)
    
    print("⚙️ Setting up git configuration...")
    
    # Set user identity
    run_command(f'git config --global user.email "{email}"')
    run_command(f'git config --global user.name "{username}"')
    
    # Set remote URL with authentication
    repo_url = f"https://{username}:{token}@github.com/richie-peters/looker-metrics.git"
    success, output = run_command(f'git remote set-url origin {repo_url}')
    
    if success:
        print("✅ Git configuration complete!")
        return True
    else:
        print(f"❌ Git configuration failed: {output}")
        return False

def create_auto_backup_decorator():
    """
    Create a decorator that automatically backs up after running functions
    """
    def auto_backup_decorator(func):
        def wrapper(*args, **kwargs):
            # Run the original function
            result = func(*args, **kwargs)
            
            # Auto-backup after function completes
            print(f"\n🔄 Auto-backup after {func.__name__}...")
            auto_git_push(
                commit_message=f"Auto-backup after running {func.__name__}",
                include_timestamp=True
            )
            
            return result
        return wrapper
    return auto_backup_decorator

# Usage examples and setup
def setup_git_auto_push():
    """Setup function - run this once"""
    print("Setting up auto-git-push...")
    
    # You'll need to provide these
    USERNAME = "richie-peters"
    EMAIL = "richie.peters@news.com.au"  
    TOKEN = "your_github_token_here"  # Replace with your actual token
    
    # Setup git config
    setup_auto_git_config(USERNAME, EMAIL, TOKEN)
    
    print("\n🎯 Auto-push functions ready to use!")
    print("\nUsage examples:")
    print("  auto_git_push()  # Simple push with default message")
    print("  auto_git_push('Fixed bug in analysis')  # Custom message")
    print("  auto_git_push(dry_run=True)  # See what would happen")

# Convenience functions
def quick_push(message="Quick update"):
    """Quick push with simple message"""
    return auto_git_push(commit_message=message)

def save_progress(description="Progress checkpoint"):
    """Save current progress"""
    return auto_git_push(commit_message=f"Progress: {description}")

def backup_now():
    """Emergency backup"""
    return auto_git_push(commit_message="Emergency backup", force_push=True)

# Example: Auto-backup decorator usage
@create_auto_backup_decorator()
def my_analysis_function():
    """Example function that auto-backs up when it finishes"""
    print("Doing some analysis...")
    # Your analysis code here
    return "Analysis complete"

print("✅ Auto-git-push functions loaded!")
print("\n🚀 To get started:")
print("1. Run: setup_git_auto_push()  # One-time setup")
print("2. Then use: auto_git_push() or quick_push('your message')")

def fix_github_authentication():
    """Fix GitHub authentication with Personal Access Token"""
    
    # You need to paste your actual token here
    USERNAME = "richie-peters"
    TOKEN = input("Paste your GitHub Personal Access Token here: ")  # This will prompt you to enter it
    
    if not TOKEN or TOKEN.strip() == "":
        print("❌ No token provided!")
        return False
    
    # Remove any whitespace
    TOKEN = TOKEN.strip()
    
    # Set the remote URL with token authentication
    import subprocess
    
    repo_url = f"https://{USERNAME}:{TOKEN}@github.com/richie-peters/looker-metrics.git"
    
    try:
        # Update the remote URL
        result = subprocess.run(
            f'git remote set-url origin {repo_url}', 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd='/content/looker-metrics'
        )
        
        if result.returncode == 0:
            print("✅ GitHub authentication updated successfully!")
            
            # Test the connection
            test_result = subprocess.run(
                'git remote -v', 
                shell=True, 
                capture_output=True, 
                text=True, 
                cwd='/content/looker-metrics'
            )
            
            if "https://" in test_result.stdout:
                print("✅ Remote URL configured correctly")
                return True
            else:
                print("⚠️ Remote URL might not be set correctly")
                return False
        else:
            print(f"❌ Failed to set remote URL: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error setting up authentication: {e}")
        return False


def check_git_status():
    """Check detailed git status"""
    import subprocess
    import os
    
    os.chdir('/content/looker-metrics')
    
    print("=== CURRENT GIT STATUS ===")
    
    # Check working directory status
    result = subprocess.run('git status', shell=True, capture_output=True, text=True)
    print("Working directory status:")
    print(result.stdout)
    
    # Check if there are unpushed commits
    print("\n=== UNPUSHED COMMITS ===")
    result = subprocess.run('git log origin/main..HEAD --oneline', shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        print("Unpushed commits:")
        print(result.stdout)
    else:
        print("No unpushed commits")
    
    # Check recent commits
    print("\n=== RECENT COMMITS ===")
    result = subprocess.run('git log --oneline -5', shell=True, capture_output=True, text=True)
    print(result.stdout)


def push_existing_commits():
    """Push any existing unpushed commits"""
    import subprocess
    import os
    
    os.chdir('/content/looker-metrics')
    
    # Check for unpushed commits
    result = subprocess.run('git log origin/main..HEAD --oneline', shell=True, capture_output=True, text=True)
    
    if result.stdout.strip():
        print("📤 Found unpushed commits, pushing now...")
        push_result = subprocess.run('git push origin main', shell=True, capture_output=True, text=True)
        
        if push_result.returncode == 0:
            print("✅ Successfully pushed existing commits!")
            return True
        else:
            print(f"❌ Push failed: {push_result.stderr}")
            return False
    else:
        print("✅ No unpushed commits found")
        return True


def fix_github_authentication():
    """Fix GitHub authentication with Personal Access Token"""
    
    # You need to paste your actual token here
    USERNAME = "richie-peters"
    TOKEN = input("Paste your GitHub Personal Access Token here: ")  # This will prompt you to enter it
    
    if not TOKEN or TOKEN.strip() == "":
        print("❌ No token provided!")
        return False
    
    # Remove any whitespace
    TOKEN = TOKEN.strip()
    
    # Set the remote URL with token authentication
    import subprocess
    
    repo_url = f"https://{USERNAME}:{TOKEN}@github.com/richie-peters/looker-metrics.git"
    
    try:
        # Update the remote URL
        result = subprocess.run(
            f'git remote set-url origin {repo_url}', 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd='/content/looker-metrics'
        )
        
        if result.returncode == 0:
            print("✅ GitHub authentication updated successfully!")
            
            # Test the connection
            test_result = subprocess.run(
                'git remote -v', 
                shell=True, 
                capture_output=True, 
                text=True, 
                cwd='/content/looker-metrics'
            )
            
            if "https://" in test_result.stdout:
                print("✅ Remote URL configured correctly")
                return True
            else:
                print("⚠️ Remote URL might not be set correctly")
                return False
        else:
            print(f"❌ Failed to set remote URL: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error setting up authentication: {e}")
        return False