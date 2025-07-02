"""
Consolidated Functions Module (With Looker Analysis Prompt)
===========================================================

This module contains all custom functions for the Looker Studio analysis pipeline,
including SQL utilities, batch processing utilities, and data processing utilities.

Author: Data Team
Date: 2025-01-07
"""

import json
import time
import os
import sys
from datetime import datetime
from google.cloud import bigquery, storage
from google.cloud import aiplatform_v1beta1, aiplatform_v1
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import vertexai
from vertexai.generative_models import GenerativeModel
import pandas as pd
import numpy as np

# Looker Analysis Prompt
LOOKER_ANALYSIS_PROMPT = """
Analyze these Looker Studio dashboard SQL queries and extract comprehensive metrics information with unified dataset analysis.
INPUT DATA:
- Dashboard ID: {dashboard_id}
- Dashboard Name: {dashboard_name}
- SQL Samples: {sql_samples}

ANALYSIS REQUIREMENTS:

1. **METRIC EXTRACTION**: Identify all business metrics, dimensions, and calculations.
2. **SQL DECOMPOSITION**: Break down complex nested queries into logical components.
3. **DEPENDENCY MAPPING**: Identify which calculations depend on others.
4. **BUSINESS LOGIC**: Extract the core business rules and transformations.
5. **DATASET STRUCTURE**: Understand data grain, key dimensions, and metric interactions.
6. **HARDCODED VALUES**: Identify hardcoded dates, values, and variables that should be parameterised or use governed tables.
7. **UNIFIED ANALYSIS**: Create consolidated queries that analyze all metrics together.
8. **STANDARDISATION**: Use standardised classes and values for consistent analysis.
9. **BIGQUERY COMPLIANCE**: Generate only valid BigQuery SQL syntax.

OUTPUT REQUIREMENTS (JSON):
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
    "primary_analysis_sql": "**THIS IS THE MAIN SQL TO RUN** - Single query showing all key metrics calculated together with appropriate sampling and date filters.",
    "structure_sql": "Query to understand data structure, grain, and key dimensions with sampling.",
    "validation_sql": "Quick validation that all metric calculations work syntactically.",
    "business_rules_sql": "Query to validate key business logic, filters, and data quality."
  }},
  "metrics": [
    {{
      "metric_id": "unique_identifier_snake_case",
      "metric_name": "Human Readable Name",
      "metric_type": "dimension|measure|calculated_field|filter|aggregation|ratio|percentage",
      "business_description": "what this metric represents in business terms",
      "sql_logic": "core SQL calculation logic extracted from queries",
      "is_kpi": "true|false",
      "business_criticality": "high|medium|low",
      "depends_on_metrics": ["metric_id1", "metric_id2"],
      "governance_issues": [
        {{
            "issue_type": "hardcoded_date|hardcoded_value|logic_concern",
            "description": "e.g., 'Hardcoded fiscal year 2025' or 'CASE statement uses hardcoded region codes'",
            "value_found": "'2025' or \"'TA', 'DT'\"",
            "recommendation": "e.g., 'Replace with date parameter' or 'Join to masthead_lookup table'"
        }}
      ],
      "data_quality_concerns": ["potential nulls", "outliers expected", "data freshness dependent"]
    }}
  ]
}}

CRITICAL BIGQUERY SQL REQUIREMENTS:

**NEVER DO THESE - THEY CAUSE FAILURES:**
- Compare INT64 with STRING: `WHERE fiscal_week_id = 'CP'`
- Use string functions on non-string types: `SUBSTR(fiscal_year, 3, 2)` if `fiscal_year` is INT64.
- Invent column names. Only use column names explicitly found in the provided SQL samples.
- Assume table structures. The provided SQL is the only source of truth for table and column names.
- Create arrays with NULLs: `ARRAY[col1, col2, null_col]`
- Missing GROUP BY: `SELECT customer, revenue FROM table GROUP BY customer`

**ALWAYS DO THESE - THEY WORK:**
- **Aggressively use SAFE_CAST on all columns involved in comparisons, functions, or joins.**
- Cast before comparing: `WHERE SAFE_CAST(fiscal_week_id AS STRING) = 'CP' OR WHERE fiscal_week_id = CAST('2024' AS INT64)`
- Cast before using string functions: `SUBSTR(SAFE_CAST(fiscal_year AS STRING), 3, 2)`
- Handle NULLs in arrays: `ARRAY(SELECT x FROM UNNEST([col1, col2, col3]) AS x WHERE x IS NOT NULL)`
- Aggregate or group all non-aggregated columns in the SELECT statement.
- **Verify column existence**: Only use columns that are present in the sample SQL queries provided.
"""



# SQL Utilities
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

def format_emails_for_sql(email_list):
    """Format email list for use in SQL IN clauses."""
    if not email_list:
        return "''"
    formatted_emails = "', '".join(email_list)
    return f"'{formatted_emails}'"

# Vertex AI Batch Processing Utilities
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


def convert_batch_results_to_dataset(results):
    """Convert batch prediction results to structured dataset - Updated for Standardised Structure"""
    try:
        # Extract responses from batch results
        responses = []
        for i, result in enumerate(results):
            try:
                # Navigate to the actual Gemini response text
                if 'response' in result and 'candidates' in result['response']:
                    candidates = result['response']['candidates']
                    if candidates and len(candidates) > 0:
                        candidate = candidates[0]
                        if 'content' in candidate and 'parts' in candidate['content']:
                            parts = candidate['content']['parts']
                            if parts and len(parts) > 0:
                                response_text = parts[0]['text']
                                responses.append({
                                    'raw_response': response_text,
                                    'status': 'success',
                                    'response_id': i
                                })
                            else:
                                print(f"No parts found in response {i}")
                        else:
                            print(f"No content/parts in response {i}")
                    else:
                        print(f"No candidates in response {i}")
                else:
                    print(f"No response/candidates in result {i}")
                    
            except Exception as e:
                print(f"Error processing result {i}: {e}")
                responses.append({
                    'raw_response': str(result),
                    'status': 'error',
                    'response_id': i
                })
        
        print(f"Extracted {len(responses)} responses")
        
        # Parse JSON responses and flatten into separate datasets
        dashboard_data = []
        metrics_data = []
        metric_interactions_data = []
        dataset_analysis_data = []
        hardcoded_issues_data = []
        
        for response in responses:
            try:
                response_text = response['raw_response']
                
                # Extract JSON from markdown code blocks
                if '```json' in response_text:
                    json_start = response_text.find('```json') + 7
                    json_end = response_text.find('```', json_start)
                    json_text = response_text[json_start:json_end].strip()
                elif '{' in response_text and '}' in response_text:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    json_text = response_text[json_start:json_end]
                else:
                    print(f"No JSON found in response {response['response_id']}")
                    continue
                
                # Parse JSON
                parsed_response = json.loads(json_text)
                
                # Extract dashboard summary
                if 'dashboard_summary' in parsed_response:
                    dashboard_summary = parsed_response['dashboard_summary'].copy()
                    dashboard_summary['response_id'] = response['response_id']
                    
                    # Process primary_data_sources (split semicolon-separated values)
                    if 'primary_data_sources' in dashboard_summary:
                        if isinstance(dashboard_summary['primary_data_sources'], str):
                            dashboard_summary['data_sources_list'] = dashboard_summary['primary_data_sources'].split(';')
                            dashboard_summary['data_sources_count'] = len(dashboard_summary['data_sources_list'])
                        else:
                            dashboard_summary['data_sources_list'] = dashboard_summary['primary_data_sources']
                            dashboard_summary['data_sources_count'] = len(dashboard_summary['primary_data_sources']) if dashboard_summary['primary_data_sources'] else 0
                    
                    # Process hardcoded dates and values (convert to counts for analysis)
                    if 'hardcoded_dates_found' in dashboard_summary:
                        dashboard_summary['hardcoded_dates_count'] = len(dashboard_summary['hardcoded_dates_found']) if dashboard_summary['hardcoded_dates_found'] else 0
                    if 'hardcoded_values_found' in dashboard_summary:
                        dashboard_summary['hardcoded_values_count'] = len(dashboard_summary['hardcoded_values_found']) if dashboard_summary['hardcoded_values_found'] else 0
                    if 'governance_opportunities' in dashboard_summary:
                        dashboard_summary['governance_opportunities_count'] = len(dashboard_summary['governance_opportunities']) if dashboard_summary['governance_opportunities'] else 0
                    
                    dashboard_data.append(dashboard_summary)
                
                # Extract dataset analysis
                if 'dataset_analysis' in parsed_response:
                    dataset_analysis = parsed_response['dataset_analysis'].copy()
                    dataset_analysis['response_id'] = response['response_id']
                    dataset_analysis['dashboard_id'] = parsed_response.get('dashboard_summary', {}).get('dashboard_id', '')
                    
                    # Extract hardcoded issues for separate analysis
                    if 'hardcoded_issues' in dataset_analysis:
                        hardcoded_issues = dataset_analysis['hardcoded_issues']
                        
                        # Process hardcoded dates
                        if 'hardcoded_dates' in hardcoded_issues:
                            for date_issue in hardcoded_issues['hardcoded_dates']:
                                date_issue_row = date_issue.copy()
                                date_issue_row['response_id'] = response['response_id']
                                date_issue_row['dashboard_id'] = parsed_response.get('dashboard_summary', {}).get('dashboard_id', '')
                                date_issue_row['issue_type'] = 'hardcoded_date'
                                hardcoded_issues_data.append(date_issue_row)
                        
                        # Process hardcoded variables
                        if 'hardcoded_variables' in hardcoded_issues:
                            for var_issue in hardcoded_issues['hardcoded_variables']:
                                var_issue_row = var_issue.copy()
                                var_issue_row['response_id'] = response['response_id']
                                var_issue_row['dashboard_id'] = parsed_response.get('dashboard_summary', {}).get('dashboard_id', '')
                                var_issue_row['issue_type'] = 'hardcoded_variable'
                                # Process hardcoded_values list
                                if 'hardcoded_values' in var_issue_row:
                                    var_issue_row['hardcoded_values_count'] = len(var_issue_row['hardcoded_values']) if var_issue_row['hardcoded_values'] else 0
                                    var_issue_row['hardcoded_values_text'] = ', '.join(var_issue_row['hardcoded_values']) if var_issue_row['hardcoded_values'] else ''
                                hardcoded_issues_data.append(var_issue_row)
                    
                    dataset_analysis_data.append(dataset_analysis)
                
                # Extract metrics
                if 'metrics' in parsed_response:
                    for metric in parsed_response['metrics']:
                        metric_row = metric.copy()
                        metric_row['response_id'] = response['response_id']
                        metric_row['dashboard_id'] = parsed_response.get('dashboard_summary', {}).get('dashboard_id', '')
                        metric_row['dashboard_name'] = parsed_response.get('dashboard_summary', {}).get('dashboard_name', '')
                        
                        # Process list fields for easier analysis
                        if 'depends_on_metrics' in metric_row:
                            metric_row['depends_on_metrics_count'] = len(metric_row['depends_on_metrics']) if metric_row['depends_on_metrics'] else 0
                            metric_row['depends_on_metrics_text'] = ', '.join(metric_row['depends_on_metrics']) if metric_row['depends_on_metrics'] else ''
                        
                        if 'data_sources' in metric_row:
                            metric_row['data_sources_count'] = len(metric_row['data_sources']) if metric_row['data_sources'] else 0
                            metric_row['data_sources_text'] = ', '.join(metric_row['data_sources']) if metric_row['data_sources'] else ''
                        
                        if 'filters_applied' in metric_row:
                            metric_row['filters_applied_count'] = len(metric_row['filters_applied']) if metric_row['filters_applied'] else 0
                            metric_row['filters_applied_text'] = ', '.join(metric_row['filters_applied']) if metric_row['filters_applied'] else ''
                        
                        if 'hardcoded_dates_in_metric' in metric_row:
                            metric_row['hardcoded_dates_in_metric_count'] = len(metric_row['hardcoded_dates_in_metric']) if metric_row['hardcoded_dates_in_metric'] else 0
                        
                        if 'hardcoded_values_in_metric' in metric_row:
                            metric_row['hardcoded_values_in_metric_count'] = len(metric_row['hardcoded_values_in_metric']) if metric_row['hardcoded_values_in_metric'] else 0
                        
                        if 'governance_issues' in metric_row:
                            metric_row['governance_issues_count'] = len(metric_row['governance_issues']) if metric_row['governance_issues'] else 0
                            metric_row['governance_issues_text'] = ', '.join(metric_row['governance_issues']) if metric_row['governance_issues'] else ''
                        
                        if 'data_quality_concerns' in metric_row:
                            metric_row['data_quality_concerns_count'] = len(metric_row['data_quality_concerns']) if metric_row['data_quality_concerns'] else 0
                            metric_row['data_quality_concerns_text'] = ', '.join(metric_row['data_quality_concerns']) if metric_row['data_quality_concerns'] else ''
                        
                        metrics_data.append(metric_row)
                
                # Extract metric interactions
                if 'metric_interactions' in parsed_response:
                    for interaction in parsed_response['metric_interactions']:
                        interaction_row = interaction.copy()
                        interaction_row['response_id'] = response['response_id']
                        interaction_row['dashboard_id'] = parsed_response.get('dashboard_summary', {}).get('dashboard_id', '')
                        
                        # Process related_metrics list
                        if 'related_metrics' in interaction_row:
                            interaction_row['related_metrics_count'] = len(interaction_row['related_metrics']) if interaction_row['related_metrics'] else 0
                            interaction_row['related_metrics_text'] = ', '.join(interaction_row['related_metrics']) if interaction_row['related_metrics'] else ''
                        
                        metric_interactions_data.append(interaction_row)
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error for response {response['response_id']}: {e}")
                print(f"Response preview: {response['raw_response'][:200]}...")
            except Exception as e:
                print(f"Error processing response {response['response_id']}: {e}")
        
        # Create DataFrames
        dashboards_df = pd.DataFrame(dashboard_data) if dashboard_data else pd.DataFrame()
        metrics_df = pd.DataFrame(metrics_data) if metrics_data else pd.DataFrame()
        metric_interactions_df = pd.DataFrame(metric_interactions_data) if metric_interactions_data else pd.DataFrame()
        dataset_analysis_df = pd.DataFrame(dataset_analysis_data) if dataset_analysis_data else pd.DataFrame()
        hardcoded_issues_df = pd.DataFrame(hardcoded_issues_data) if hardcoded_issues_data else pd.DataFrame()
        
        print(f"✓ Created datasets:")
        print(f"  - Dashboards: {len(dashboards_df)} rows")
        print(f"  - Metrics: {len(metrics_df)} rows")
        print(f"  - Metric Interactions: {len(metric_interactions_df)} rows")
        print(f"  - Dataset Analysis: {len(dataset_analysis_df)} rows")
        print(f"  - Hardcoded Issues: {len(hardcoded_issues_df)} rows")
        
        return {
            'dashboards': dashboards_df,
            'metrics': metrics_df,
            'metric_interactions': metric_interactions_df,
            'dataset_analysis': dataset_analysis_df,
            'hardcoded_issues': hardcoded_issues_df,
            'raw_responses': pd.DataFrame(responses)
        }
        
    except Exception as e:
        print(f"✗ Failed to convert results: {e}")
        return None


# Legacy function for compatibility
def run_gemini_batch_fast(requests, project, display_name, input_gcs_uri,
                         output_gcs_uri, model_name="gemini-2.5-flash",
                         location="us-central1", max_output_tokens=65000,
                         temperature=0.20):
    """Legacy function - redirects to slick version"""
    return run_gemini_batch_fast_slick(requests, project, display_name, input_gcs_uri,
                                      output_gcs_uri, model_name, location, max_output_tokens, temperature)

def prepare_looker_analysis_batch(df):
    """Convert dataframe to batch input format with structured Looker analysis prompt."""
    batch_data = []
    
    for dashboard_id, group in df.groupby('looker_studio_report_id'):
        dashboard_data = {
            "dashboard_id": dashboard_id,
            "dashboard_name": group.iloc[0]['looker_studio_report_name'],
            "dashboard_owner": group.iloc[0]['assetOwner'],
            "sql_samples": []
        }
        
        for _, row in group.iterrows():
            dashboard_data["sql_samples"].append({
                "job_id": row['jobId'],
                "username": row['username'],
                "runtime_seconds": row['runtime_seconds'],
                "total_processed_bytes": row['totalProcessedBytes'] if pd.notna(row['totalProcessedBytes']) else None,  # Handle NaN
                "sql_query": row['query_text']
            })
        
        # Format the structured prompt for this dashboard
        formatted_prompt = LOOKER_ANALYSIS_PROMPT.format(
            dashboard_id=dashboard_data["dashboard_id"],
            dashboard_name=dashboard_data["dashboard_name"],
            sql_samples=json.dumps(dashboard_data["sql_samples"], indent=2, default=str)  # Add default=str
        )
        
        batch_data.append({"content": formatted_prompt})
    
    return batch_data


def save_datasets_to_csv(datasets, output_folder="./data/"):
    """Save datasets to CSV files"""
    try:
        import os
        os.makedirs(output_folder, exist_ok=True)

        # Save each dataset
        for name, df in datasets.items():
            if df is not None and len(df) > 0:
                output_path = f"{output_folder}looker_analysis_{name}.csv"
                df.to_csv(output_path, index=False)
                print(f"✓ Saved {name}: {output_path} ({len(df)} rows)")
            else:
                print(f"⚠️ Skipped {name}: empty dataset")
        
        return True 
        
    except Exception as e:
        print(f"✗ Failed to save datasets: {e}")
        return False

def analyse_results_summary(datasets):
    """Quick analysis of the results"""
    print("\n" + "="*60)
    print("LOOKER ANALYSIS RESULTS SUMMARY")
    print("="*60)
    
    if 'dashboards' in datasets and len(datasets['dashboards']) > 0:
        dashboards_df = datasets['dashboards']
        print(f"\n📊 DASHBOARDS ANALYSED: {len(dashboards_df)}")
        
        if 'business_domain' in dashboards_df.columns:
            print("\nDomains:")
            print(dashboards_df['business_domain'].value_counts())
        
        if 'complexity_score' in dashboards_df.columns:
            print(f"\nComplexity scores:")
            print(f"  Average: {dashboards_df['complexity_score'].mean():.1f}")
            print(f"  Range: {dashboards_df['complexity_score'].min()} - {dashboards_df['complexity_score'].max()}")
    
    if 'metrics' in datasets and len(datasets['metrics']) > 0:
        metrics_df = datasets['metrics']
        print(f"\n📈 METRICS IDENTIFIED: {len(metrics_df)}")
        
        if 'metric_type' in metrics_df.columns:
            print("\nMetric types:")
            print(metrics_df['metric_type'].value_counts())
        
        if 'calculation_type' in metrics_df.columns:
            print("\nCalculation types:")
            print(metrics_df['calculation_type'].value_counts())
        
        if 'is_final_output' in metrics_df.columns:
            final_outputs = len(metrics_df[metrics_df['is_final_output'] == True])
            print(f"\nFinal output metrics: {final_outputs}")
    
    if 'raw_responses' in datasets and len(datasets['raw_responses']) > 0:
        raw_df = datasets['raw_responses']
        successful = len(raw_df[raw_df['status'] == 'success'])
        failed = len(raw_df[raw_df['status'] == 'error'])
        print(f"\n📈 PROCESSING SUMMARY:")
        print(f"  Successful responses: {successful}")
        print(f"  Failed responses: {failed}")
    
    print("\n" + "="*60)


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

# Example usage function
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



def troubleshoot_failed_queries(execution_results):
    """Analyze and categorize failed queries to improve the prompt"""
    
    failed_queries = execution_results['metadata'][execution_results['metadata']['execution_status'] == 'failed']
    
    if failed_queries.empty:
        print("No failed queries to troubleshoot!")
        return
    
    print("=== FAILED QUERY ANALYSIS ===\n")
    
    error_categories = {
        'type_mismatch': [],
        'function_signature': [],
        'group_by_issues': [],
        'array_issues': [],
        'other': []
    }
    
    for _, row in failed_queries.iterrows():
        error_msg = row['error_message'].lower()
        
        if 'no matching signature for operator' in error_msg and ('int64' in error_msg or 'string' in error_msg):
            error_categories['type_mismatch'].append(row)
        elif 'no matching signature for function' in error_msg:
            error_categories['function_signature'].append(row)
        elif 'neither grouped nor' in error_msg or 'group by' in error_msg:
            error_categories['group_by_issues'].append(row)
        elif 'array cannot have a null element' in error_msg:
            error_categories['array_issues'].append(row)
        else:
            error_categories['other'].append(row)
    
    # Print categorized errors
    for category, errors in error_categories.items():
        if errors:
            print(f"\n{category.upper().replace('_', ' ')} ({len(errors)} errors):")
            for error in errors:
                print(f"  Dashboard: {error['dashboard_id'][:8]}...")
                print(f"  Query Type: {error['query_type']}")
                print(f"  Error: {error['error_message'][:150]}...")
                print()
    
    return error_categories

def get_failed_sql_for_inspection(dataset_df, execution_results, error_type='type_mismatch'):
    """Get the actual SQL from failed queries for inspection"""
    
    failed_metadata = execution_results['metadata'][execution_results['metadata']['execution_status'] == 'failed']
    
    print(f"=== FAILED SQL INSPECTION ({error_type.upper()}) ===\n")
    
    for _, row in failed_metadata.head(2).iterrows():  # Show first 2 failed queries
        dashboard_id = row['dashboard_id']
        query_type = row['query_type']
        
        # Get the SQL from original dataset
        original_row = dataset_df[dataset_df['dashboard_id'] == dashboard_id].iloc[0]
        sql = original_row[query_type]
        
        print(f"Dashboard: {dashboard_id}")
        print(f"Query Type: {query_type}")
        print(f"Error: {row['error_message']}")
        print("\nSQL:")
        print("-" * 80)
        print(sql[:500] + "..." if len(sql) > 500 else sql)
        print("-" * 80)
        print("\n")

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

def create_unified_dataset_from_existing(datasets, df_bq_results=None):
    """
    Create unified dataset from your existing separate datasets
    """
    import pandas as pd
    
    print("🔗 CREATING UNIFIED DATASET FROM YOUR EXISTING DATA")
    print("=" * 55)
    
    # Get your datasets
    dashboards_df = datasets.get('dashboards', pd.DataFrame())
    metrics_df = datasets.get('metrics', pd.DataFrame()) 
    metric_interactions_df = datasets.get('metric_interactions', pd.DataFrame())
    dataset_analysis_df = datasets.get('dataset_analysis', pd.DataFrame())
    hardcoded_issues_df = datasets.get('hardcoded_issues', pd.DataFrame())
    
    print(f"Input datasets:")
    print(f"  Dashboards: {len(dashboards_df)} rows")
    print(f"  Metrics: {len(metrics_df)} rows") 
    print(f"  Metric Interactions: {len(metric_interactions_df)} rows")
    print(f"  Dataset Analysis: {len(dataset_analysis_df)} rows")
    print(f"  Hardcoded Issues: {len(hardcoded_issues_df)} rows")
    
    unified_records = []
    
    # Create dashboard-level records
    print("\n📊 Creating dashboard-level records...")
    for _, dashboard in dashboards_df.iterrows():
        dashboard_id = dashboard['dashboard_id']
        
        # Get related data for this dashboard
        dashboard_metrics = metrics_df[metrics_df['dashboard_id'] == dashboard_id] if not metrics_df.empty else pd.DataFrame()
        dashboard_interactions = metric_interactions_df[metric_interactions_df['dashboard_id'] == dashboard_id] if not metric_interactions_df.empty else pd.DataFrame()
        dashboard_analysis = dataset_analysis_df[dataset_analysis_df['dashboard_id'] == dashboard_id] if not dataset_analysis_df.empty else pd.DataFrame()
        dashboard_issues = hardcoded_issues_df[hardcoded_issues_df['dashboard_id'] == dashboard_id] if not hardcoded_issues_df.empty else pd.DataFrame()
        
        # Create dashboard summary record  
        dashboard_record = {
            # Core identifiers
            'record_id': f"{dashboard_id}_dashboard",
            'dashboard_id': dashboard_id,
            'response_id': dashboard.get('response_id', ''),
            'record_type': 'dashboard_summary',
            
            # Dashboard info
            'dashboard_name': dashboard.get('dashboard_name', ''),
            'business_domain': dashboard.get('business_domain', ''),
            'complexity_score': dashboard.get('complexity_score', 0),
            'consolidation_score': dashboard.get('consolidation_score', 0),
            'date_grain': dashboard.get('date_grain', ''),
            'data_grain': dashboard.get('data_grain', ''),
            'primary_data_sources': dashboard.get('primary_data_sources', ''),
            'date_range_detected': dashboard.get('date_range_detected', ''),
            
            # Aggregated metrics info
            'total_metrics_count': len(dashboard_metrics),
            'kpi_metrics_count': len(dashboard_metrics[dashboard_metrics.get('is_kpi', False) == True]) if not dashboard_metrics.empty else 0,
            'final_output_metrics_count': len(dashboard_metrics[dashboard_metrics.get('is_final_output', False) == True]) if not dashboard_metrics.empty else 0,
            
            # Governance summary
            'hardcoded_dates_count': dashboard.get('hardcoded_dates_count', 0),
            'hardcoded_values_count': dashboard.get('hardcoded_values_count', 0),
            'governance_issues_count': len(dashboard_issues),
            
            # Interactions
            'metric_interactions_count': len(dashboard_interactions),
            
            # Text summaries for embeddings
            'dashboard_description': create_dashboard_description(dashboard, dashboard_metrics),
            'metrics_summary': create_metrics_summary_text(dashboard_metrics),
            'governance_summary': create_governance_summary_text(dashboard_issues),
            'full_context': 'dashboard_summary'
        }
        
        unified_records.append(dashboard_record)
        
        # Create individual metric records
        for _, metric in dashboard_metrics.iterrows():
            metric_record = {
                'record_id': f"{dashboard_id}_{metric.get('metric_id', 'unknown')}",
                'dashboard_id': dashboard_id,
                'response_id': metric.get('response_id', ''),
                'record_type': 'metric',
                
                # Link to dashboard
                'dashboard_name': dashboard.get('dashboard_name', ''),
                'business_domain': dashboard.get('business_domain', ''),
                
                # Metric details
                'metric_id': metric.get('metric_id', ''),
                'metric_name': metric.get('metric_name', ''),
                'metric_type': metric.get('metric_type', ''),
                'calculation_type': metric.get('calculation_type', ''),
                'is_kpi': metric.get('is_kpi', False),
                'is_final_output': metric.get('is_final_output', False),
                'business_criticality': metric.get('business_criticality', ''),
                'metric_category': metric.get('metric_category', ''),
                'business_description': metric.get('business_description', ''),
                'sql_logic': metric.get('sql_logic', ''),
                
                # Dependencies and relationships
                'depends_on_metrics_count': metric.get('depends_on_metrics_count', 0),
                'data_sources_count': metric.get('data_sources_count', 0),
                'governance_issues_count': metric.get('governance_issues_count', 0),
                
                # Text description
                'dashboard_description': create_metric_description(metric, dashboard),
                'metrics_summary': metric.get('business_description', ''),
                'governance_summary': metric.get('governance_issues_text', ''),
                'full_context': 'individual_metric'
            }
            
            unified_records.append(metric_record)
    
    # Convert to DataFrame
    unified_df = pd.DataFrame(unified_records)
    
    print(f"\n✅ Created unified dataset:")
    print(f"   Total records: {len(unified_df)}")
    print(f"   Dashboard summaries: {len(unified_df[unified_df['record_type'] == 'dashboard_summary'])}")
    print(f"   Individual metrics: {len(unified_df[unified_df['record_type'] == 'metric'])}")
    print(f"   Columns: {len(unified_df.columns)}")
    
    return unified_df

def create_dashboard_description(dashboard, metrics):
    """Create readable description of dashboard"""
    parts = []
    
    name = dashboard.get('dashboard_name', 'Unknown Dashboard')
    domain = dashboard.get('business_domain', 'unknown')
    
    parts.append(f"{name} is a {domain} dashboard")
    
    if not metrics.empty:
        kpis = len(metrics[metrics.get('is_kpi', False) == True])
        total = len(metrics)
        parts.append(f"containing {total} metrics")
        if kpis > 0:
            parts.append(f"including {kpis} key performance indicators")
    
    complexity = dashboard.get('complexity_score', 0)
    if complexity > 7:
        parts.append("with high analytical complexity")
    elif complexity > 4:
        parts.append("with moderate complexity")
    
    sources = dashboard.get('primary_data_sources', '')
    if sources:
        source_count = len(sources.split(';'))
        parts.append(f"drawing from {source_count} data sources")
    
    return '. '.join(parts) + '.'

def create_metrics_summary_text(metrics):
    """Create text summary of metrics"""
    if metrics.empty:
        return "No metrics defined."
    
    parts = []
    total = len(metrics)
    kpis = len(metrics[metrics.get('is_kpi', False) == True])
    
    parts.append(f"Contains {total} metrics")
    if kpis > 0:
        parts.append(f"{kpis} are key performance indicators")
    
    if 'metric_category' in metrics.columns:
        categories = metrics['metric_category'].value_counts()
        if not categories.empty:
            top_cat = categories.index[0]
            parts.append(f"primarily focused on {top_cat} metrics")
    
    return '. '.join(parts) + '.'

def create_governance_summary_text(issues):
    """Create governance issues summary"""
    if issues.empty:
        return "No governance issues identified."
    
    parts = []
    total = len(issues)
    parts.append(f"Has {total} governance issues")
    
    if 'issue_type' in issues.columns:
        issue_types = issues['issue_type'].value_counts()
        for issue_type, count in issue_types.head(2).items():
            parts.append(f"{count} {issue_type.replace('_', ' ')} issues")
    
    return '. '.join(parts) + '.'

def create_metric_description(metric, dashboard):
    """Create description for individual metric"""
    parts = []
    
    name = metric.get('metric_name', 'Unknown Metric')
    mtype = metric.get('metric_type', 'unknown')
    calc_type = metric.get('calculation_type', 'unknown')
    
    parts.append(f"{name} is a {mtype} metric using {calc_type} calculation")
    
    if metric.get('is_kpi'):
        parts.append("classified as a key performance indicator")
    
    if metric.get('business_description'):
        parts.append(f"measuring {metric['business_description']}")
    
    parts.append(f"from the {dashboard.get('dashboard_name', 'unknown')} dashboard")
    
    return '. '.join(parts) + '.'

def prepare_secondary_batch_input_clean(unified_dataset, df_bq_results):
    """
    Clean version - only use dashboard summary records
    """
    import json
    
    print("📊 PREPARING SECONDARY BATCH INPUT (CLEAN VERSION)")
    print("=" * 55)
    
    # Filter to only dashboard summary records and remove NaN columns
    dashboard_summaries = unified_dataset[
        unified_dataset['record_type'] == 'dashboard_summary'
    ].dropna(axis=1, how='all')
    
    print(f"Processing {len(dashboard_summaries)} dashboard summaries")
    
    secondary_batch_data = []
    
    # Get the combined BQ results
    combined_results = df_bq_results.get('combined_results', {})
    
    for _, dashboard in dashboard_summaries.iterrows():
        dashboard_id = dashboard['dashboard_id']
        
        print(f"  Processing: {dashboard.get('dashboard_name', 'Unknown')[:50]}...")
        
        # Get original analysis data - only use non-NaN values
        original_analysis = {}
        for key in ['dashboard_name', 'business_domain', 'complexity_score', 'consolidation_score', 
                   'total_metrics_count', 'primary_data_sources', 'date_grain', 'data_grain',
                   'kpi_metrics_count', 'governance_issues_count']:
            value = dashboard.get(key)
            if pd.notna(value):
                original_analysis[key] = value
        
        original_analysis['dashboard_id'] = dashboard_id
        
        # Get actual SQL results for this dashboard
        actual_results = {}
        
        for query_type in ['primary_analysis_sql', 'structure_sql', 'validation_sql', 'business_rules_sql']:
            if query_type in combined_results and combined_results[query_type] is not None:
                dashboard_data = combined_results[query_type][
                    combined_results[query_type]['dashboard_id'] == dashboard_id
                ]
                
                if not dashboard_data.empty:
                    # Convert to JSON-serializable format
                    actual_results[query_type] = {
                        'row_count': len(dashboard_data),
                        'columns': [col for col in dashboard_data.columns if col not in ['dashboard_id', 'response_id', 'query_type']],
                        'sample_data': dashboard_data.head(3).to_dict('records'),
                        'data_summary': f"Dataset contains {len(dashboard_data)} rows with {len(dashboard_data.columns)} columns"
                    }
                else:
                    actual_results[query_type] = {'row_count': 0, 'message': 'No data returned'}
            else:
                actual_results[query_type] = {'message': 'Query not executed or failed'}
        
        # Get related metrics for this dashboard from the unified dataset
        dashboard_metrics = unified_dataset[
            (unified_dataset['dashboard_id'] == dashboard_id) & 
            (unified_dataset['record_type'] == 'metric')
        ]
        
        metrics_info = []
        for _, metric in dashboard_metrics.iterrows():
            metric_dict = {}
            for key in ['metric_id', 'metric_name', 'metric_type', 'business_description', 
                       'sql_logic', 'is_kpi', 'calculation_type', 'metric_category']:
                value = metric.get(key)
                if pd.notna(value):
                    metric_dict[key] = value
            if metric_dict:  # Only add if we have some data
                metrics_info.append(metric_dict)
        
        # Create the input data structure
        secondary_input_data = {
            'dashboard_id': dashboard_id,
            'dashboard_name': dashboard.get('dashboard_name', ''),
            'original_dashboard_analysis': original_analysis,
            'dashboard_metrics': metrics_info,
            'actual_sql_results': actual_results
        }
        
        # Get the secondary prompt
        secondary_prompt = design_secondary_analysis_prompt()
        
        # Format the prompt with the data
        try:
            formatted_prompt = secondary_prompt.format(
                dashboard_id=dashboard_id,
                dashboard_name=dashboard.get('dashboard_name', ''),
                original_dashboard_analysis=json.dumps(original_analysis, indent=2),
                actual_sql_results=json.dumps(actual_results, indent=2),
                primary_analysis_data=json.dumps(actual_results.get('primary_analysis_sql', {}), indent=2),
                structure_analysis_data=json.dumps(actual_results.get('structure_sql', {}), indent=2),
                validation_results=json.dumps(actual_results.get('validation_sql', {}), indent=2),
                business_rules_data=json.dumps(actual_results.get('business_rules_sql', {}), indent=2)
            )
            
            secondary_batch_data.append({
                'content': formatted_prompt,
                'dashboard_id': dashboard_id,
                'dashboard_name': dashboard.get('dashboard_name', '')
            })
            
        except Exception as e:
            print(f"    ⚠️ Error formatting prompt for {dashboard_id}: {e}")
            continue
    
    print(f"✅ Successfully prepared {len(secondary_batch_data)} secondary analysis requests")
    return secondary_batch_data


def design_secondary_analysis_prompt():
    """
    Designs the secondary analysis prompt to validate initial findings
    and propose a data-backed consolidation strategy.
    """
    secondary_prompt = """
    SECONDARY LOOKER CONSOLIDATION ANALYSIS: DATA VALIDATION & STRATEGY
    =====================================================================

    You are an expert data architect. Your task is to analyze an initial AI assessment of a Looker Studio dashboard against **actual data results** from executed BigQuery queries. Your goal is to verify the initial findings and propose a high-level, data-backed strategy for consolidation.

    **INPUT DATA:**
    - Dashboard ID: {dashboard_id}
    - Dashboard Name: {dashboard_name}
    - Initial AI-Generated Analysis: {original_dashboard_analysis}
    - **Actual Data from Primary Analysis Query**: {primary_analysis_data}
    - **Actual Data from Structure Analysis Query**: {structure_analysis_data}
    - **Actual Data from Validation Query**: {validation_results}
    - **Actual Data from Business Rules Query**: {business_rules_data}

    **CORE OBJECTIVES:**
    1.  **VERIFY METRICS**: Use the provided data samples to confirm or challenge the initial metric definitions.
    2.  **IDENTIFY CONSOLIDATION OPPORTUNITIES**: Based on the actual data, identify metrics or data sources that are clear candidates for consolidation.
    3.  **FORMULATE STRATEGY**: Propose a high-level strategy for how to approach the consolidation for this dashboard.

    **OUTPUT REQUIREMENTS (JSON):**
    {{
      "consolidation_analysis": {{
        "dashboard_id": "string",
        "dashboard_name": "string",
        "consolidation_priority": "high|medium|low (Based on complexity and opportunities found)",
        "migration_complexity": "1-10 (Based on the number of sources, metrics, and data quality issues found)",
        "data_quality_score": "1-10 (Based on nulls, inconsistencies, or failed validation queries in the provided data)",
        "key_findings_from_data": "A summary of the most important insights derived directly from analyzing the query results."
      }},
      "metrics_consolidation": [
        {{
          "current_metric_name": "string",
          "consolidation_target_metric": "A proposed unified metric name (e.g., 'unified_revenue_usd').",
          "data_backed_rationale": "Explain WHY these metrics should be consolidated, using evidence from the 'actual_sql_results'. For example: 'The data shows that metric_A from this dashboard and metric_B from another dashboard have nearly identical value ranges and patterns, suggesting they measure the same business concept.'",
          "proposed_transformation": "A high-level description of the transformation logic needed (e.g., 'COALESCE values and convert currency')."
        }}
      ],
      "data_source_mapping": [
        {{
          "current_source": "project.dataset.table",
          "consolidation_target": "proposed new unified table/view",
          "mapping_complexity": "1-10",
          "data_backed_rationale": "Explain WHY this mapping is proposed, using evidence from the 'structure_analysis_data' (e.g., 'This source contains transactional data with a grain that matches the proposed unified `transactions` table.')."
        }}
      ],
      "consolidation_strategy": {{
          "strategic_summary": "A plain-english summary of how to approach consolidating this dashboard.",
          "key_prerequisites": ["List of cleanup tasks or dependencies that must be addressed first (e.g., 'Address hardcoded values in `metric_X`', 'Create unified `customer_lookup` table')."],
          "proposed_sequencing": ["High-level order of operations (e.g., '1. Unify data sources. 2. Consolidate revenue metrics. 3. Decommission old views.')"]
      }},
      "english_summary": {{
        "dashboard_story": "A simple, one-paragraph description of what this dashboard does for business users.",
        "consolidation_story": "A simple, one-paragraph explanation of how this dashboard fits into the larger consolidation effort and the benefits of doing so."
      }}
    }}
    """
    return secondary_prompt


def prepare_secondary_batch_input_robust(unified_dataset, df_bq_results):
    """
    Robust version - handles dates, NaNs, and missing data gracefully
    """
    import json
    import pandas as pd
    from datetime import date, datetime
    import numpy as np
    
    print("📊 PREPARING SECONDARY BATCH INPUT (ROBUST VERSION)")
    print("=" * 55)
    
    def clean_for_json(obj):
        """Clean data for JSON serialization"""
        if pd. isna(obj) or obj is None:
            return None
        elif isinstance(obj, (date, datetime)):
            return obj.strftime('%Y-%m-%d') if hasattr(obj, 'strftime') else str(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if not pd.isna(obj) else None
        elif isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        else:
            return obj
    
    # Filter to dashboard summaries only
    dashboard_summaries = unified_dataset[
        unified_dataset['record_type'] == 'dashboard_summary'
    ]
    
    print(f"Processing {len(dashboard_summaries)} dashboard summaries")
    
    secondary_batch_data = []
    combined_results = df_bq_results.get('combined_results', {})
    
    for _, dashboard in dashboard_summaries.iterrows():
        dashboard_id = dashboard['dashboard_id']
        dashboard_name = dashboard.get('dashboard_name', 'Unknown Dashboard')
        
        print(f"  Processing: {dashboard_name[:50]}...")
        
        try:
            # Create original analysis with cleaned data
            original_analysis = {
                'dashboard_id': dashboard_id,
                'dashboard_name': clean_for_json(dashboard.get('dashboard_name')),
                'business_domain': clean_for_json(dashboard.get('business_domain')),
                'complexity_score': clean_for_json(dashboard.get('complexity_score')),
                'consolidation_score': clean_for_json(dashboard.get('consolidation_score')),
                'total_metrics_count': clean_for_json(dashboard.get('total_metrics_count')),
                'kpi_metrics_count': clean_for_json(dashboard.get('kpi_metrics_count')),
                'data_grain': clean_for_json(dashboard.get('data_grain')),
                'date_grain': clean_for_json(dashboard.get('date_grain')),
                'governance_issues_count': clean_for_json(dashboard.get('governance_issues_count'))
            }
            
            # Remove None values
            original_analysis = {k: v for k, v in original_analysis.items() if v is not None}
            
            # Get SQL results - proceed even if some are missing
            actual_results = {}
            
            for query_type in ['primary_analysis_sql', 'structure_sql', 'validation_sql', 'business_rules_sql']:
                try:
                    if query_type in combined_results and combined_results[query_type] is not None:
                        dashboard_data = combined_results[query_type][
                            combined_results[query_type]['dashboard_id'] == dashboard_id
                        ]
                        
                        if not dashboard_data.empty:
                            # Get a small sample and clean it
                            sample_data = dashboard_data.head(3).copy()
                            
                            # Clean sample data for JSON serialization
                            cleaned_sample = []
                            for _, row in sample_data.iterrows():
                                cleaned_row = {}
                                for col, val in row.items():
                                    if col not in ['dashboard_id', 'response_id', 'query_type']:
                                        cleaned_val = clean_for_json(val)
                                        if cleaned_val is not None:
                                            cleaned_row[col] = cleaned_val
                                if cleaned_row:  # Only add if we have some data
                                    cleaned_sample.append(cleaned_row)
                            
                            actual_results[query_type] = {
                                'status': 'success',
                                'row_count': len(dashboard_data),
                                'columns': [col for col in dashboard_data.columns 
                                          if col not in ['dashboard_id', 'response_id', 'query_type']],
                                'sample_data': cleaned_sample[:2],  # Just 2 rows to keep prompt manageable
                                'data_summary': f"Contains {len(dashboard_data)} rows with business data"
                            }
                        else:
                            actual_results[query_type] = {
                                'status': 'no_data',
                                'message': 'Query executed but returned no rows for this dashboard'
                            }
                    else:
                        actual_results[query_type] = {
                            'status': 'not_available',
                            'message': 'Query not executed or failed - will analyse based on available data'
                        }
                except Exception as e:
                    actual_results[query_type] = {
                        'status': 'error',
                        'message': f'Error processing query results: {str(e)[:100]}'
                    }
            
            # Get metrics for this dashboard
            dashboard_metrics = unified_dataset[
                (unified_dataset['dashboard_id'] == dashboard_id) & 
                (unified_dataset['record_type'] == 'metric')
            ]
            
            metrics_info = []
            for _, metric in dashboard_metrics.iterrows():
                metric_dict = {}
                for key in ['metric_id', 'metric_name', 'metric_type', 'business_description', 
                           'is_kpi', 'calculation_type', 'metric_category']:
                    value = clean_for_json(metric.get(key))
                    if value is not None:
                        metric_dict[key] = value
                
                if len(metric_dict) > 2:  # Only include if we have meaningful data
                    metrics_info.append(metric_dict)
            
            # Create the complete input structure
            secondary_input = {
                'dashboard_id': dashboard_id,
                'dashboard_name': dashboard_name,
                'original_dashboard_analysis': original_analysis,
                'dashboard_metrics': metrics_info,
                'actual_sql_results': actual_results,
                'data_availability': {
                    'has_primary_analysis': actual_results.get('primary_analysis_sql', {}).get('status') == 'success',
                    'has_structure_analysis': actual_results.get('structure_sql', {}).get('status') == 'success',
                    'has_validation_results': actual_results.get('validation_sql', {}).get('status') == 'success',
                    'has_business_rules': actual_results.get('business_rules_sql', {}).get('status') == 'success',
                    'metrics_count': len(metrics_info)
                }
            }
            
            # Get the secondary prompt
            secondary_prompt = design_secondary_analysis_prompt()
            
            # Format the prompt - this should work now with cleaned data
            formatted_prompt = secondary_prompt.format(
                dashboard_id=dashboard_id,
                dashboard_name=dashboard_name,
                original_dashboard_analysis=json.dumps(original_analysis, indent=2),
                actual_sql_results=json.dumps(actual_results, indent=2),
                primary_analysis_data=json.dumps(actual_results.get('primary_analysis_sql', {}), indent=2),
                structure_analysis_data=json.dumps(actual_results.get('structure_sql', {}), indent=2),
                validation_results=json.dumps(actual_results.get('validation_sql', {}), indent=2),
                business_rules_data=json.dumps(actual_results.get('business_rules_sql', {}), indent=2)
            )
            
            secondary_batch_data.append({
                'content': formatted_prompt,
                'dashboard_id': dashboard_id,
                'dashboard_name': dashboard_name,
                'data_quality': secondary_input['data_availability']
            })
            
            print(f"    ✅ Successfully prepared request")
            
        except Exception as e:
            print(f"    ⚠️ Error preparing {dashboard_id}: {str(e)[:100]}")
            continue
    
    print(f"\n✅ Successfully prepared {len(secondary_batch_data)} secondary analysis requests")
    
    # Show data quality summary
    if secondary_batch_data:
        print(f"\n📊 DATA QUALITY SUMMARY:")
        total_with_primary = sum(1 for req in secondary_batch_data if req['data_quality']['has_primary_analysis'])
        total_with_structure = sum(1 for req in secondary_batch_data if req['data_quality']['has_structure_analysis'])
        total_with_validation = sum(1 for req in secondary_batch_data if req['data_quality']['has_validation_results'])
        
        print(f"  Requests with primary analysis data: {total_with_primary}/{len(secondary_batch_data)}")
        print(f"  Requests with structure analysis data: {total_with_structure}/{len(secondary_batch_data)}")
        print(f"  Requests with validation data: {total_with_validation}/{len(secondary_batch_data)}")
        print(f"  Average metrics per dashboard: {sum(req['data_quality']['metrics_count'] for req in secondary_batch_data) / len(secondary_batch_data):.1f}")
    
    return secondary_batch_data


def extract_secondary_results_to_datasets(secondary_results):
    """
    Extract secondary analysis results into structured datasets
    
    Args:
        secondary_results: The results from your secondary batch analysis
        
    Returns:
        dict: Dictionary containing multiple DataFrames for different aspects
    """
    import pandas as pd
    import json
    
    print("🔍 EXTRACTING SECONDARY ANALYSIS RESULTS")
    print("=" * 50)
    
    # Extract responses from batch results
    responses = []
    for i, result in enumerate(secondary_results):
        try:
            if 'response' in result and 'candidates' in result['response']:
                candidates = result['response']['candidates']
                if candidates and len(candidates) > 0:
                    candidate = candidates[0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if parts and len(parts) > 0:
                            response_text = parts[0]['text']
                            responses.append({
                                'raw_response': response_text,
                                'status': 'success',
                                'response_id': i
                            })
                        else:
                            print(f"No parts found in response {i}")
                    else:
                        print(f"No content/parts in response {i}")
                else:
                    print(f"No candidates in response {i}")
            else:
                print(f"No response/candidates in result {i}")
        except Exception as e:
            print(f"Error processing result {i}: {e}")
            responses.append({
                'raw_response': str(result),
                'status': 'error',
                'response_id': i
            })
    
    print(f"Extracted {len(responses)} responses")
    
    # Initialize datasets
    consolidation_analysis_data = []
    metrics_consolidation_data = []
    data_source_mapping_data = []
    relationship_model_data = []
    transformation_specs_data = []
    migration_plan_data = []
    english_summaries_data = []
    quality_assessment_data = []
    
    # Process each response
    for response in responses:
        try:
            response_text = response['raw_response']
            
            # Extract JSON from markdown code blocks
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                json_text = response_text[json_start:json_end].strip()
            elif '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_text = response_text[json_start:json_end]
            else:
                print(f"No JSON found in response {response['response_id']}")
                continue
            
            # Parse JSON
            parsed_response = json.loads(json_text)
            response_id = response['response_id']
            
            # Extract consolidation_analysis
            if 'consolidation_analysis' in parsed_response:
                analysis = parsed_response['consolidation_analysis'].copy()
                analysis['response_id'] = response_id
                consolidation_analysis_data.append(analysis)
            
            # Extract metrics_consolidation (array)
            if 'metrics_consolidation' in parsed_response:
                for metric in parsed_response['metrics_consolidation']:
                    metric_row = metric.copy()
                    metric_row['response_id'] = response_id
                    metric_row['dashboard_id'] = parsed_response.get('consolidation_analysis', {}).get('dashboard_id', '')
                    
                    # Process arrays in metrics consolidation
                    if 'similar_metrics_across_dashboards' in metric_row:
                        metric_row['similar_metrics_count'] = len(metric_row['similar_metrics_across_dashboards']) if metric_row['similar_metrics_across_dashboards'] else 0
                        metric_row['similar_metrics_text'] = ', '.join(metric_row['similar_metrics_across_dashboards']) if metric_row['similar_metrics_across_dashboards'] else ''
                    
                    if 'data_quality_issues' in metric_row:
                        metric_row['data_quality_issues_count'] = len(metric_row['data_quality_issues']) if metric_row['data_quality_issues'] else 0
                        metric_row['data_quality_issues_text'] = ', '.join(metric_row['data_quality_issues']) if metric_row['data_quality_issues'] else ''
                    
                    metrics_consolidation_data.append(metric_row)
            
            # Extract data_source_mapping (array)
            if 'data_source_mapping' in parsed_response:
                for mapping in parsed_response['data_source_mapping']:
                    mapping_row = mapping.copy()
                    mapping_row['response_id'] = response_id
                    mapping_row['dashboard_id'] = parsed_response.get('consolidation_analysis', {}).get('dashboard_id', '')
                    
                    # Process arrays in data source mapping
                    if 'data_quality_concerns' in mapping_row:
                        mapping_row['data_quality_concerns_count'] = len(mapping_row['data_quality_concerns']) if mapping_row['data_quality_concerns'] else 0
                        mapping_row['data_quality_concerns_text'] = ', '.join(mapping_row['data_quality_concerns']) if mapping_row['data_quality_concerns'] else ''
                    
                    if 'migration_prerequisites' in mapping_row:
                        mapping_row['migration_prerequisites_count'] = len(mapping_row['migration_prerequisites']) if mapping_row['migration_prerequisites'] else 0
                        mapping_row['migration_prerequisites_text'] = ', '.join(mapping_row['migration_prerequisites']) if mapping_row['migration_prerequisites'] else ''
                    
                    # Convert key_fields_mapping dict to string
                    if 'key_fields_mapping' in mapping_row and isinstance(mapping_row['key_fields_mapping'], dict):
                        mapping_row['key_fields_mapping_text'] = ', '.join([f"{k}->{v}" for k, v in mapping_row['key_fields_mapping'].items()])
                    
                    data_source_mapping_data.append(mapping_row)
            
            # Extract relationship_model
            if 'relationship_model' in parsed_response:
                model = parsed_response['relationship_model'].copy()
                model['response_id'] = response_id
                model['dashboard_id'] = parsed_response.get('consolidation_analysis', {}).get('dashboard_id', '')
                
                # Process arrays in relationship model
                for array_field in ['current_relationships', 'key_entities', 'relationship_changes', 'consolidation_benefits', 'implementation_challenges']:
                    if array_field in model:
                        model[f'{array_field}_count'] = len(model[array_field]) if model[array_field] else 0
                        model[f'{array_field}_text'] = ', '.join(model[array_field]) if model[array_field] else ''
                
                relationship_model_data.append(model)
            
            # Extract transformation_specifications (array)
            if 'transformation_specifications' in parsed_response:
                for transform in parsed_response['transformation_specifications']:
                    transform_row = transform.copy()
                    transform_row['response_id'] = response_id
                    transform_row['dashboard_id'] = parsed_response.get('consolidation_analysis', {}).get('dashboard_id', '')
                    transformation_specs_data.append(transform_row)
            
            # Extract migration_plan
            if 'migration_plan' in parsed_response:
                plan = parsed_response['migration_plan'].copy()
                plan['response_id'] = response_id
                plan['dashboard_id'] = parsed_response.get('consolidation_analysis', {}).get('dashboard_id', '')
                
                # Process arrays in migration plan
                for array_field in ['prerequisites', 'migration_steps', 'testing_phases', 'rollback_triggers', 'business_validation_required', 'go_live_criteria', 'post_migration_monitoring']:
                    if array_field in plan:
                        plan[f'{array_field}_count'] = len(plan[array_field]) if plan[array_field] else 0
                        plan[f'{array_field}_text'] = ', '.join(plan[array_field]) if plan[array_field] else ''
                
                migration_plan_data.append(plan)
            
            # Extract english_summaries
            if 'english_summaries' in parsed_response:
                summary = parsed_response['english_summaries'].copy()
                summary['response_id'] = response_id
                summary['dashboard_id'] = parsed_response.get('consolidation_analysis', {}).get('dashboard_id', '')
                english_summaries_data.append(summary)
            
            # Extract quality_assessment
            if 'quality_assessment' in parsed_response:
                quality = parsed_response['quality_assessment'].copy()
                quality['response_id'] = response_id
                quality['dashboard_id'] = parsed_response.get('consolidation_analysis', {}).get('dashboard_id', '')
                
                # Process arrays in quality assessment
                for array_field in ['data_accuracy_issues', 'calculation_validation_results', 'business_logic_issues', 'performance_concerns', 'scalability_issues', 'recommendations']:
                    if array_field in quality:
                        quality[f'{array_field}_count'] = len(quality[array_field]) if quality[array_field] else 0
                        quality[f'{array_field}_text'] = ', '.join(quality[array_field]) if quality[array_field] else ''
                
                quality_assessment_data.append(quality)
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error for response {response['response_id']}: {e}")
        except Exception as e:
            print(f"Error processing response {response['response_id']}: {e}")
    
    # Create DataFrames
    datasets = {
        'consolidation_analysis': pd.DataFrame(consolidation_analysis_data) if consolidation_analysis_data else pd.DataFrame(),
        'metrics_consolidation': pd.DataFrame(metrics_consolidation_data) if metrics_consolidation_data else pd.DataFrame(),
        'data_source_mapping': pd.DataFrame(data_source_mapping_data) if data_source_mapping_data else pd.DataFrame(),
        'relationship_model': pd.DataFrame(relationship_model_data) if relationship_model_data else pd.DataFrame(),
        'transformation_specifications': pd.DataFrame(transformation_specs_data) if transformation_specs_data else pd.DataFrame(),
        'migration_plan': pd.DataFrame(migration_plan_data) if migration_plan_data else pd.DataFrame(),
        'english_summaries': pd.DataFrame(english_summaries_data) if english_summaries_data else pd.DataFrame(),
        'quality_assessment': pd.DataFrame(quality_assessment_data) if quality_assessment_data else pd.DataFrame(),
        'raw_responses': pd.DataFrame(responses)
    }
    
    # Print summary
    print(f"\n✅ EXTRACTION COMPLETE:")
    for name, df in datasets.items():
        if len(df) > 0:
            print(f"  - {name}: {len(df)} rows, {len(df.columns)} columns")
        else:
            print(f"  - {name}: empty dataset")
    
    return datasets

# Quick analysis function
def analyze_secondary_results_summary(secondary_datasets):
    """Quick analysis of secondary results"""
    print("\n" + "="*60)
    print("SECONDARY ANALYSIS RESULTS SUMMARY")
    print("="*60)
    
    # Consolidation analysis summary
    if 'consolidation_analysis' in secondary_datasets and len(secondary_datasets['consolidation_analysis']) > 0:
        consolidation_df = secondary_datasets['consolidation_analysis']
        print(f"\n📊 CONSOLIDATION ANALYSIS: {len(consolidation_df)} dashboards")
        
        if 'consolidation_priority' in consolidation_df.columns:
            print("\nConsolidation Priority:")
            print(consolidation_df['consolidation_priority'].value_counts())
        
        if 'migration_complexity' in consolidation_df.columns:
            print(f"\nMigration Complexity:")
            print(f"  Average: {consolidation_df['migration_complexity'].mean():.1f}")
            print(f"  Range: {consolidation_df['migration_complexity'].min()} - {consolidation_df['migration_complexity'].max()}")
        
        if 'consolidation_readiness' in consolidation_df.columns:
            print("\nConsolidation Readiness:")
            print(consolidation_df['consolidation_readiness'].value_counts())
    
    # Metrics consolidation summary
    if 'metrics_consolidation' in secondary_datasets and len(secondary_datasets['metrics_consolidation']) > 0:
        metrics_df = secondary_datasets['metrics_consolidation']
        print(f"\n📈 METRICS CONSOLIDATION: {len(metrics_df)} consolidation opportunities")
        
        if 'business_impact' in metrics_df.columns:
            print("\nBusiness Impact:")
            print(metrics_df['business_impact'].value_counts())
        
        if 'migration_order' in metrics_df.columns:
            print(f"\nMigration Order Distribution:")
            print(metrics_df['migration_order'].value_counts().sort_index())
    
    # Data source mapping summary
    if 'data_source_mapping' in secondary_datasets and len(secondary_datasets['data_source_mapping']) > 0:
        mapping_df = secondary_datasets['data_source_mapping']
        print(f"\n🗂️ DATA SOURCE MAPPING: {len(mapping_df)} source mappings")
        
        if 'transformation_type' in mapping_df.columns:
            print("\nTransformation Types:")
            print(mapping_df['transformation_type'].value_counts())
        
        if 'mapping_complexity' in mapping_df.columns:
            print(f"\nMapping Complexity:")
            print(f"  Average: {mapping_df['mapping_complexity'].mean():.1f}")
    
    print("\n" + "="*60)

# Usage function
def save_secondary_datasets_to_csv(secondary_datasets, output_folder="./data/secondary/"):
    """Save secondary analysis datasets to CSV files"""
    import os
    
    try:
        os.makedirs(output_folder, exist_ok=True)
        
        for name, df in secondary_datasets.items():
            if df is not None and len(df) > 0:
                output_path = f"{output_folder}secondary_analysis_{name}.csv" 
                df.to_csv(output_path, index=False)
                print(f"✓ Saved {name}: {output_path} ({len(df)} rows)")
            else:
                print(f"⚠️ Skipped {name}: empty dataset")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to save datasets: {e}")
        return False
