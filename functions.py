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

# Add this import at the top of your functions.py file
import traceback
# Ensure these imports are at the top of your functions.py file
import json
import pandas as pd
import numpy as np
from datetime import date, datetime
from decimal import Decimal

# ==============================================================================
# 1. PROMPT DEFINITIONS
# ==============================================================================
# ==============================================================================
# 1. PROMPT DEFINITIONS (UPDATED)
# ==============================================================================
# ==============================================================================
# 1. PROMPT DEFINITIONS (UPDATED)
# ==============================================================================

# ==============================================================================
# 1. PROMPT DEFINITIONS (UPDATED)
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

# This is the final, highly specific prompt. (UPDATED)
METRIC_CONSOLIDATION_PROMPT = """
Act as an expert data architect. Your task is to analyze a comprehensive list of metrics from an enterprise reporting suite to find opportunities for consolidation. You must group metrics that share the same business purpose, even if their SQL implementations are slightly different.

**CRITICAL INSTRUCTIONS:**
1.  **Only report on meaningful consolidation opportunities.** If you do not find any groups of metrics that are duplicated or semantically similar, return an an empty list.
2.  Your entire output MUST be a single JSON object with one key: "consolidation_details".
3.  This key must contain a flat list of metric objects. DO NOT nest the objects.
4.  For each group you identify, repeat the 'group_' level information for every metric that belongs to that group.
5.  **Be concise and highly specific.** Focus on the technical and logical differences between metrics. Do not provide a general summary.
    - Specifically, look for and describe differences in:
        - **Date/Time Logic**: E.g., one uses `WEEK(MONDAY)` while another uses `WEEK(SUNDAY)`.
        - **Calculations**: E.g., `revenue / users` vs. `revenue / (users - returns)`.
        - **Hardcoded values**: E.g., one uses a `CASE` statement with 8 categories, another case statement has a 9th.
        - **Filters/Conditions**: E.g., one filters for 'active' users, another filters for 'active' and 'pending'.

**INPUT DATA:**
- A JSON list of ALL metric objects. The object contains the following fields: `metric_id`, `metric_name`, `metric_sql_core`, `dashboard_name`, `gcp_project_name`, `dataset_name`, and `table_name`.

{all_metrics_json}

**OUTPUT REQUIREMENTS (JSON Schema):**
Return a flat list of objects. Each object in the list must conform to the following structure:
{{
  "consolidation_id": "A unique identifier for the group (e.g., 'group_arpu').",
  "group_business_explanation": "A high-level, business-friendly summary of what this consolidated metric represents.",
  "group_variance_summary": "An overall summary of how and why the metrics in this group differ from each other.",
  "metric_id": "The unique ID of this specific metric.",
  "metric_name": "The name of this specific metric as it appears in its dashboard.",
  "dashboard_name": "The name of the dashboard where this metric is found.",
  "metric_similarity_percentage": "A score (0-100) indicating how similar this metric is to the other metrics in its group.",
  "metric_specific_difference": "A precise, technical explanation of how THIS INDIVIDUAL metric's logic differs from the rest of the group. (e.g., 'This ARPU calculation includes subscription revenue, while others do not.')."
}}
"""
# --- Start: Single-Batch Consolidation Analysis (Corrected) ---

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
    Creates a unified dataset from the primary analysis outputs.
    This version corrects the 'record_type' to align with downstream functions.
    """
    print("🔗 Creating unified dataset from analysis results...")

    # Extract the dataframes from the primary output
    dashboards_df = datasets.get('dashboards')
    metrics_df = datasets.get('metrics')
    governance_issues_df = datasets.get('governance_issues')

    if dashboards_df is None or dashboards_df.empty:
        print("⚠️ No dashboard data found to create a unified dataset.")
        return pd.DataFrame()

    # Pre-aggregate governance issues for easier lookup
    if governance_issues_df is not None and not governance_issues_df.empty:
        metric_issue_counts = governance_issues_df.groupby('dashboard_id').size().reset_index(name='total_governance_issues')
        dashboards_df = pd.merge(dashboards_df, metric_issue_counts, on='dashboard_id', how='left')
        dashboards_df['total_governance_issues'] = dashboards_df['total_governance_issues'].fillna(0).astype(int)
    else:
        dashboards_df['total_governance_issues'] = 0


    # Assemble records
    unified_records = []
    for _, dashboard in dashboards_df.iterrows():
        dashboard_id = dashboard['dashboard_id']
        
        dashboard_metrics = metrics_df[metrics_df['dashboard_id'] == dashboard_id] if metrics_df is not None else pd.DataFrame()

        # Create a summary text block for the dashboard
        metrics_summary_parts = []
        if not dashboard_metrics.empty:
            for _, metric in dashboard_metrics.head(5).iterrows(): # Summary of top 5 metrics
                metrics_summary_parts.append(f"- {metric.get('metric_name', 'Unnamed Metric')}: {metric.get('business_description', 'No description.')}")
        metrics_summary_text = "\\n".join(metrics_summary_parts) # Use \\n for literal newline in a string

        # Build the unified record for the dashboard
        dashboard_record = {
            'record_id': f"{dashboard_id}_summary",
            
            # --- THE FIX ---
            'record_type': 'dashboard_summary', # Correctly set to 'dashboard_summary'
            # ---------------------
            
            'dashboard_id': dashboard_id,
            'dashboard_name': dashboard.get('dashboard_name'),
            'business_domain': dashboard.get('business_domain'),
            'complexity_score': dashboard.get('complexity_score'),
            'consolidation_score': dashboard.get('consolidation_score'),
            'total_metrics': len(dashboard_metrics),
            'total_governance_issues': dashboard.get('total_governance_issues', 0),
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

def run_complete_analysis(dataset_df, project, max_dashboards=None, timeout_seconds=20):
    """
    Orchestrates the execution of AI-generated SQL queries with a specified timeout.
    """
    from google.cloud import bigquery
    
    client = bigquery.Client(project=project)
    print(f"Starting complete analysis for up to {max_dashboards or len(dataset_df)} dashboards with a {timeout_seconds}s timeout per query...")
    
    # Execute all queries with the timeout
    successful_results, execution_metadata_df = execute_dataset_analysis_queries(
        dataset_df=dataset_df,
        client=client, 
        project=project, 
        max_queries=max_dashboards,
        timeout_seconds=timeout_seconds
    )
    
    # Analyze the performance based on the metadata
    performance_analysis = analyze_query_performance(execution_metadata_df)
    
    print(f"\n✅ SQL Execution Stage Complete.")
    print(f"   - Attempted: {len(execution_metadata_df)} queries.")
    print(f"   - Successful: {len(successful_results)} queries.")
    
    return {
        'successful_results': successful_results,
        'execution_metadata': execution_metadata_df
    }

def execute_dataset_analysis_queries(dataset_df, client, project, max_queries=None, timeout_seconds=20):
    """
    Executes all SQL queries from the dataset analysis DataFrame with a timeout.
    """
    query_columns = ['primary_analysis_sql', 'structure_sql', 'validation_sql', 'business_rules_sql']
    successful_results = []
    all_metadata = []
    
    rows_to_process = dataset_df.head(max_queries) if max_queries else dataset_df
    print(f"Executing up to {len(rows_to_process) * len(query_columns)} queries for {len(rows_to_process)} dashboards...")

    for idx, row in rows_to_process.iterrows():
        dashboard_id = row['dashboard_id']
        response_id = row['response_id']
        print(f"\n--- Processing Dashboard {idx + 1}/{len(rows_to_process)}: {dashboard_id[:8]}... ---")

        for query_type in query_columns:
            sql = row.get(query_type)
            
            if pd.isna(sql) or not sql.strip() or "query generation skipped" in sql.lower():
                print(f"⚠️ Skipping '{query_type}': No valid SQL provided.")
                continue
            
            # Pass the timeout to the execution function
            result = execute_sql_with_metadata(
                sql=sql,
                query_type=query_type,
                dashboard_id=dashboard_id,
                response_id=response_id,
                client=client,
                project=project,
                timeout_seconds=timeout_seconds
            )
            
            all_metadata.append(result['metadata'])
            if result['data'] is not None and not result['data'].empty:
                successful_results.append(result)
    
    return successful_results, pd.DataFrame(all_metadata)

def execute_sql_with_metadata(sql, query_type, dashboard_id, response_id, client=None, project=None, timeout_seconds=60):
    """Execute a single SQL query and return results with metadata, including a job timeout."""
    import time
    from datetime import datetime
    from google.cloud import bigquery
    
    if client is None:
        client = bigquery.Client(project=project)
    
    # Configure the query job with a timeout
    job_config = bigquery.QueryJobConfig(
        # The timeout is in milliseconds
        job_timeout_ms=timeout_seconds * 1000
    )
    
    start_time = time.time()
    
    try:
        # Execute the query with the specified job configuration
        job = client.query(sql, job_config=job_config)
        results = job.result()  # Wait for the job to complete
        
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
    Prepares the batch input for the secondary, deeper analysis, combining
    the initial AI analysis with the results from the live SQL query executions.
    """
    print("📊 PREPARING SECONDARY BATCH INPUT (ROBUST VERSION)")
    print("=" * 55)

    def clean_for_json(obj):
        """Helper function to make data JSON-serializable."""
        if isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        if isinstance(obj, pd.DataFrame):
            return [clean_for_json(row) for row in obj.to_dict('records')]
        if isinstance(obj, Decimal): return float(obj)
        if isinstance(obj, (date, datetime)): return obj.isoformat()
        if pd.isna(obj) or obj is None: return None
        if isinstance(obj, np.generic): return obj.item()
        if isinstance(obj, dict): return {str(k): clean_for_json(v) for k, v in obj.items()}
        return obj

    dashboard_summaries = unified_dataset[unified_dataset['record_type'] == 'dashboard_summary']
    metrics_df = datasets.get('metrics', pd.DataFrame())
    
    print(f"Processing {len(dashboard_summaries)} dashboard summaries for secondary analysis")

    secondary_batch_data = []
    
    # Extract metadata and successful results from the BQ execution dictionary
    execution_metadata_df = df_bq_results.get('execution_metadata', pd.DataFrame())
    successful_results = df_bq_results.get('successful_results', [])

    for _, dashboard in dashboard_summaries.iterrows():
        dashboard_id = dashboard['dashboard_id']
        dashboard_name = dashboard.get('dashboard_name', 'Unknown Dashboard')
        print(f"  Processing: {dashboard_name[:50]}...")

        try:
            original_analysis = {k: v for k, v in dashboard.to_dict().items() if pd.notna(v)}

            # Get metric details for this specific dashboard
            metrics_details = []
            if not metrics_df.empty:
                dashboard_metrics = metrics_df[metrics_df['dashboard_id'] == dashboard_id]
                metrics_details = dashboard_metrics.to_dict('records')

            # Consolidate the results of the executed SQL queries for this dashboard
            sql_execution_summary = {}
            if not execution_metadata_df.empty:
                dashboard_metadata = execution_metadata_df[execution_metadata_df['dashboard_id'] == dashboard_id]
                for _, meta_row in dashboard_metadata.iterrows():
                    query_type = meta_row['query_type']
                    status = meta_row['execution_status']
                    error = meta_row.get('error_message')
                    
                    summary = {'status': status}
                    if error:
                        summary['error'] = error

                    # Find the actual data for successful queries
                    if status == 'success':
                        for res in successful_results:
                            # Match using response_id and query_type as a composite key
                            if res['metadata']['response_id'] == meta_row['response_id'] and res['metadata']['query_type'] == query_type:
                                sample_data = res.get('data')
                                if sample_data is not None:
                                    summary['sample_data'] = sample_data.head(3).to_dict('records')
                                break
                    
                    sql_execution_summary[query_type] = summary

            secondary_prompt = design_secondary_analysis_prompt()
            
            # Clean all data before serialization
            cleaned_original_analysis = clean_for_json(original_analysis)
            cleaned_metrics_details = clean_for_json(metrics_details)
            cleaned_sql_results = clean_for_json(sql_execution_summary)

            formatted_prompt = secondary_prompt.format(
                dashboard_id=dashboard_id,
                dashboard_name=dashboard_name,
                original_dashboard_analysis=json.dumps(cleaned_original_analysis, indent=2),
                metrics_details=json.dumps(cleaned_metrics_details, indent=2),
                sql_execution_summary=json.dumps(cleaned_sql_results, indent=2)
            )

            secondary_batch_data.append({'content': formatted_prompt})
            print(f"    ✅ Successfully prepared secondary request for {dashboard_name[:50]}")

        except Exception as e:
            print(f"    ⚠️ Error preparing secondary request for {dashboard_id}: {e}")
            traceback.print_exc()
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
# --- Start: Replacement code for functions.py ---

def extract_secondary_results_to_datasets(secondary_results):
    """
    Extracts the results from the secondary analysis batch job into structured
    DataFrames, aligned with the secondary prompt schema.
    """
    print("\n🔍 EXTRACTING SECONDARY ANALYSIS RESULTS")
    print("=" * 50)

    consolidation_analysis_data = []
    coding_practice_review_data = []
    investigation_points_data = []

    for response_id, result in enumerate(secondary_results):
        try:
            raw_text = result.get('response', {}).get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            if not raw_text:
                continue

            # FIX 1: Made the regex non-greedy (.*?) to better handle malformed JSON
            json_text_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
            if not json_text_match:
                json_text_match = re.search(r'(\{.*?\})', raw_text, re.DOTALL)
            
            if not json_text_match:
                continue
                
            parsed_response = json.loads(json_text_match.group(1))
            
            analysis_summary = parsed_response.get('consolidation_analysis')
            if analysis_summary:
                dashboard_id = analysis_summary.get('dashboard_id')
                analysis_summary['response_id'] = response_id
                consolidation_analysis_data.append(analysis_summary)

                if 'coding_practice_review' in parsed_response:
                    for issue in parsed_response['coding_practice_review']:
                        issue.update({'response_id': response_id, 'dashboard_id': dashboard_id})
                        coding_practice_review_data.append(issue)
                
                if 'investigation_points' in parsed_response:
                    for point in parsed_response['investigation_points']:
                        point.update({'response_id': response_id, 'dashboard_id': dashboard_id})
                        investigation_points_data.append(point)

        except Exception as e:
            print(f"✗ Error processing secondary response {response_id}: {e}")

    datasets = {
        'consolidation_analysis': pd.DataFrame(consolidation_analysis_data),
        'coding_practice_review': pd.DataFrame(coding_practice_review_data),
        'investigation_points': pd.DataFrame(investigation_points_data),
    }

    print("\n✅ Secondary extraction complete:")
    for name, df in datasets.items():
        if not df.empty:
            print(f"  - Created '{name}' with {len(df)} rows.")
    
    # FIX 2: Corrected the return variable from 'dataset' to 'datasets'
    return datasets

# --- End: Replacement code for functions.py ---

def analyze_secondary_results_summary(secondary_datasets):
    """
    Provides a quick summary analysis of the secondary analysis results.
    """
    print("\n" + "="*60)
    print("📊 SECONDARY ANALYSIS RESULTS SUMMARY")
    print("="*60)

    if 'consolidation_analysis' in secondary_datasets and not secondary_datasets['consolidation_analysis'].empty:
        consolidation_df = secondary_datasets['consolidation_analysis']
        print(f"\n📋 CONSOLIDATION ANALYSIS: {len(consolidation_df)} dashboards reviewed")
        
        if 'consolidation_priority' in consolidation_df.columns:
            print("\nConsolidation Priority:")
            print(consolidation_df['consolidation_priority'].value_counts().to_string())
        
        if 'data_verifiability_score' in consolidation_df.columns:
            print(f"\nData Verifiability Score (1-10):")
            print(f"  Average: {consolidation_df['data_verifiability_score'].mean():.1f}")

    if 'coding_practice_review' in secondary_datasets and not secondary_datasets['coding_practice_review'].empty:
        coding_review_df = secondary_datasets['coding_practice_review']
        print(f"\n\n💻 CODING PRACTICE REVIEW: {len(coding_review_df)} issues identified")
        
        if 'issue_type' in coding_review_df.columns:
            print("\nIssue Types Found:")
            print(coding_review_df['issue_type'].value_counts().to_string())
    
    print("\n" + "="*60)


def save_secondary_datasets_to_csv(secondary_datasets, output_folder="./data/secondary/"):
    """
    Saves the secondary analysis datasets to a dedicated 'secondary' folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    if not isinstance(secondary_datasets, dict) or not secondary_datasets:
        print("⚠️ No secondary datasets provided to save.")
        return False

    print(f"\n💾 Saving {len(secondary_datasets)} secondary dataset(s) to '{output_folder}'...")
    try:
        for name, df in secondary_datasets.items():
            if df is not None and not df.empty:
                output_path = os.path.join(output_folder, f"secondary_analysis_{name}.csv")
                df.to_csv(output_path, index=False)
                print(f"✓ Saved {name}: {output_path} ({len(df)} rows)")
            else:
                print(f"⚠️ Skipped saving '{name}': The dataset was empty.")
        return True
    except Exception as e:
        print(f"✗ Failed to save secondary datasets: {e}")
        return False



# --- Start: Replacement code for functions.py ---

def run_metric_consolidation_analysis(project, input_gcs_uri, output_gcs_uri, model_name):
    """
    Runs a single-batch consolidation analysis that provides all metrics to the
    model at once, leveraging the new, concise 'metric_sql_core' field.
    """
    print("\n--- Starting Single-Batch Metric Consolidation Analysis ---")

    # 1. Read and prepare the data
    try:
        metrics_df = pd.read_csv("./data/looker_analysis_metrics.csv")
        dashboards_df = pd.read_csv("./data/looker_analysis_dashboards.csv")
        
        # Merge metrics with dashboard data to get the business domain
        metrics_df = pd.merge(metrics_df, dashboards_df[['dashboard_id', 'dashboard_name', 'business_domain']], on='dashboard_id', how='left')
        
        # Ensure we have the required columns from the initial analysis
        required_cols = ['metric_id', 'metric_name', 'sql_logic', 'metric_sql_core', 'business_description', 'dashboard_id', 'dashboard_name', 'gcp_project_name', 'dataset_name', 'table_name', 'business_domain']
        for col in required_cols:
            if col not in metrics_df.columns:
                print(f"✗ Missing required column '{col}'. Please ensure your initial analysis output includes this field.")
                return None
        
        if metrics_df.empty:
            print(" Metrics dataset is empty. Skipping consolidation analysis.")
            return None
            
        print(f"✓ Loaded {len(metrics_df)} metrics for analysis.")
    
    except FileNotFoundError:
        print("✗ Could not find necessary CSV files. Please ensure 'looker_analysis_metrics.csv' and 'looker_analysis_dashboards.csv' exist.")
        return None

    # 2. Prepare the single, consolidated input payload for the model
    print("\n--- Preparing a single batch input for Gemini ---")
    
    # CORRECTED: Only send the most essential fields to prevent token overflow.
    metrics_to_analyze = metrics_df[['metric_id', 'metric_name', 'metric_sql_core', 'dashboard_name', 'gcp_project_name', 'dataset_name', 'table_name']].to_dict('records')
    
    all_metrics_json = json.dumps(metrics_to_analyze, indent=2)

    formatted_prompt = METRIC_CONSOLIDATION_PROMPT.format(all_metrics_json=all_metrics_json)
    
    consolidation_requests = [{'content': formatted_prompt}]
    
    # 3. Run the single batch job
    consolidation_results = run_gemini_batch_fast_slick(
        requests=consolidation_requests,
        project=project,
        display_name="looker-consolidation-full-batch",
        input_gcs_uri=input_gcs_uri.replace('.jsonl', '_full_batch.jsonl'),
        output_gcs_uri=output_gcs_uri.rstrip('/') + '_full_batch/',
        model_name=model_name
    )
    
    if not consolidation_results:
        print("✗ Full batch consolidation analysis failed.")
        return None
        
    # 4. Parse the single response and save
    try:
        raw_text = consolidation_results[0].get('response', {}).get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
        json_text_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
        if not json_text_match:
            json_text_match = re.search(r'(\{.*?\})', raw_text, re.DOTALL)
        
        parsed_response = json.loads(json_text_match.group(1))
        
        consolidation_details = parsed_response.get('consolidation_details', [])
        
        if not consolidation_details:
            print("✅ Analysis complete. No significant consolidation opportunities were identified by the AI.")
            return None

        details_df = pd.DataFrame(consolidation_details)
        
        output_folder = "./data/secondary/"
        os.makedirs(output_folder, exist_ok=True)
        details_path = os.path.join(output_folder, "secondary_analysis_consolidation_details.csv")
        
        details_df.to_csv(details_path, index=False)
        
        group_count = details_df['consolidation_id'].nunique()
        metric_count = len(details_df)
        print(f"\n✓ Successfully identified {metric_count} metrics for consolidation into {group_count} unique groups.")
        print(f"✓ Saved final detailed analysis to: {details_path}")
        
        return details_df

    except Exception as e:
        print(f"✗ Failed to parse final consolidation results: {e}")
        return None

# --- End: Single-Batch Consolidation Analysis (Corrected) ---