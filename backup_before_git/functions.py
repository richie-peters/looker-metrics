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

1. **METRIC EXTRACTION**: Identify all business metrics, dimensions, and calculations
2. **SQL DECOMPOSITION**: Break down complex nested queries into logical components
3. **DEPENDENCY MAPPING**: Identify which calculations depend on others
4. **BUSINESS LOGIC**: Extract the core business rules and transformations
5. **DATASET STRUCTURE**: Understand data grain, key dimensions, and metric interactions
6. **HARDCODED VALUES**: Identify hardcoded dates, values, and variables that should be parameterised or use governed tables
7. **UNIFIED ANALYSIS**: Create consolidated queries that analyze all metrics together
8. **STANDARDISATION**: Use standardised classes and values for consistent analysis
9. **BIGQUERY COMPLIANCE**: Generate only valid BigQuery SQL syntax

OUTPUT REQUIREMENTS (JSON):
Return a flat JSON structure with the following schema:

{{
  "dashboard_summary": {{
    "dashboard_id": "string",
    "dashboard_name": "string",
    "primary_data_sources": "project1.dataset1.table1;project2.dataset2.table2;project3.dataset3.table3",
    "business_domain": "advertising|finance|consumer|operations|marketing|sales|product|hr|other",
    "complexity_score": 1-10,
    "consolidation_score": 1-10,
    "total_metrics_identified": number,
    "date_grain": "daily|weekly|monthly|quarterly|yearly|mixed|none",
    "data_grain": "transactional|aggregate",
    "key_dimensions": ["date", "customer_id", "product", "region"],
    "date_range_detected": "01/01/2024 to Current",
    "hardcoded_dates_found": ["03/07/2024", "30/03/2025", "01/01/2024"],
    "hardcoded_values_found": ["specific product IDs", "region codes", "business unit names"],
    "governance_opportunities": ["dates should be parameterised", "product codes should join to product_master", "regions should use location_hierarchy"]
  }},
  "dataset_analysis": {{
    "primary_analysis_sql": "**THIS IS THE MAIN SQL TO RUN** - Single query showing all key metrics calculated together with appropriate sampling and date filters",
    "structure_sql": "Query to understand data structure, grain, and key dimensions with sampling",
    "validation_sql": "Quick validation that all metric calculations work syntactically",
    "business_rules_sql": "Query to validate key business logic, filters, and data quality",
    "sample_data_sql": "Query to get representative sample data for further analysis",
    "hardcoded_issues": {{
      "hardcoded_dates": [
        {{
          "date_value": "03/07/2024",
          "original_format": "2024-07-03T00:00:00",
          "context": "used as baseline date in DATETIME_DIFF calculation",
          "suggested_fix": "replace with CURRENT_DATE() or parameter",
          "impact": "high|medium|low",
          "urgency": "high|medium|low"
        }}
      ],
      "hardcoded_variables": [
        {{
          "variable_type": "lookup_codes|business_rules|thresholds|categories|other",
          "hardcoded_values": ["'TA'", "'DT'", "'HS'"],
          "context": "masthead codes hardcoded in CASE statement",
          "suggested_governance": "join to masthead_lookup table",
          "impact": "high|medium|low",
          "maintenance_risk": "high|medium|low"
        }}
      ]
    }},
    "parameterisation_recommendations": [
      "Replace hardcoded dates with date parameters or relative date functions",
      "Replace hardcoded lookup values with joins to governed reference tables",
      "Use configuration tables for business rules instead of hardcoded logic"
    ]
  }},
  "metrics": [
    {{
      "metric_id": "unique_identifier_snake_case",
      "metric_name": "Human Readable Name",
      "metric_type": "dimension|measure|calculated_field|filter|aggregation|ratio|percentage",
      "calculation_type": "sum|count|count_distinct|average|min|max|ratio|case_when|date_function|string_function|mathematical|conditional",
      "data_type": "numeric|string|date|boolean|array",
      "aggregation_level": "transaction|daily|weekly|monthly|quarterly|yearly|customer|product|region|custom",
      "is_final_output": true|false,
      "is_kpi": true|false,
      "business_criticality": "high|medium|low",
      "depends_on_metrics": ["metric_id1", "metric_id2"],
      "business_description": "what this metric represents in business terms",
      "sql_logic": "core SQL calculation logic extracted from queries",
      "data_sources": ["project.dataset.table1", "project.dataset.table2"],
      "filters_applied": ["date filters", "business rules", "exclusions"],
      "expected_data_type": "integer|decimal|string|date|boolean",
      "business_context": "how this metric fits into overall business analysis",
      "metric_category": "revenue|cost|volume|efficiency|quality|growth|retention|acquisition|other",
      "update_frequency": "real_time|hourly|daily|weekly|monthly|quarterly|yearly|on_demand",
      "seasonality_impact": "high|medium|low|none",
      "hardcoded_dates_in_metric": ["03/07/2024", "01/01/2025"],
      "hardcoded_values_in_metric": ["'Metro'", "'Regional'", "'TA'", "'DT'"],
      "governance_issues": ["date should be parameterised", "lookup values should use reference table"],
      "data_quality_concerns": ["potential nulls", "outliers expected", "data freshness dependent"]
    }}
  ],
  "metric_interactions": [
    {{
      "interaction_type": "mathematical_relationship|dependency|filter_impact|hierarchical|causal",
      "primary_metric": "metric_id",
      "related_metrics": ["metric_id1", "metric_id2"],
      "relationship_description": "how these metrics relate to each other",
      "mathematical_formula": "if applicable: A = B * C or A = B + C",
      "business_validation": "what this relationship means for business analysis",
      "validation_sql": "SQL to test this relationship holds true",
      "relationship_strength": "strong|medium|weak",
      "business_impact": "high|medium|low"
    }}
  ]
}}

CRITICAL BIGQUERY SQL REQUIREMENTS:

**NEVER DO THESE - THEY CAUSE FAILURES:**
❌ Compare INT64 with STRING: WHERE fiscal_week_id = 'CP'
❌ Use LPAD on INT64: LPAD(week_number, 2, '0')
❌ Create arrays with NULLs: ARRAY[col1, col2, null_col]
❌ Missing GROUP BY: SELECT customer, revenue FROM table GROUP BY customer
❌ Compare different types: WHERE year_column = 'CP' (if year_column is INT64)

**ALWAYS DO THESE - THEY WORK:**
✅ Cast before comparing: WHERE CAST(fiscal_week_id AS STRING) = 'CP' OR WHERE fiscal_week_id = CAST('2024' AS INT64)
✅ Cast before string functions: LPAD(CAST(week_number AS STRING), 2, '0')
✅ Handle NULLs in arrays: ARRAY(SELECT x FROM UNNEST([col1, col2, col3]) AS x WHERE x IS NOT NULL)
✅ Aggregate or group all columns: SELECT customer, SUM(revenue) as total_revenue FROM table GROUP BY customer
✅ Use SAFE_CAST for safety: WHERE SAFE_CAST(column AS STRING) = 'value'

DATASET ANALYSIS REQUIREMENTS (BIGQUERY-COMPLIANT):

1. **primary_analysis_sql**: **THE MAIN QUERY TO EXECUTE**
   ```sql
   -- Example structure - MUST be valid BigQuery syntax:
   WITH base_data AS (
     SELECT 
       date_dimension,
       primary_grouping_dimension,
       revenue_column,
       customer_column
     FROM `project.dataset.table`
     WHERE SAFE_CAST(date_column AS DATE) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
   )
   SELECT 
     date_dimension,
     primary_grouping_dimension,
     COUNT(*) as record_count,
     SUM(SAFE_CAST(revenue_column AS NUMERIC)) as total_revenue,
     COUNT(DISTINCT customer_column) as unique_customers,
     AVG(SAFE_CAST(revenue_column AS NUMERIC)) as avg_revenue_per_record
   FROM base_data
   GROUP BY date_dimension, primary_grouping_dimension  -- MUST include all non-aggregated columns
   ORDER BY date_dimension DESC
   LIMIT 100


structure_sql: Understand data structure - HANDLE TYPE MISMATCHES
SELECT 
  'Data Structure Analysis' as analysis_type,
  COUNT(*) as total_records,
  COUNT(DISTINCT SAFE_CAST(date_column AS DATE)) as unique_dates,
  COUNT(DISTINCT customer_column) as unique_customers,
  COUNT(DISTINCT SAFE_CAST(fiscal_week_id AS STRING)) as unique_fiscal_weeks,
  MIN(SAFE_CAST(date_column AS DATE)) as earliest_date,
  MAX(SAFE_CAST(date_column AS DATE)) as latest_date,
  APPROX_COUNT_DISTINCT(primary_key) as approx_unique_records
FROM `project.dataset.main_table` 
WHERE SAFE_CAST(date_column AS DATE) >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)



validation_sql: Quick validation - USE SAFE_CAST
SELECT 
  'Validation Check' as test_type,
  CASE WHEN SUM(SAFE_CAST(revenue AS NUMERIC)) > 0 THEN 'PASS' ELSE 'FAIL' END as revenue_test,
  CASE WHEN COUNT(DISTINCT customer_column) > 0 THEN 'PASS' ELSE 'FAIL' END as customer_test,
  CASE WHEN MAX(SAFE_CAST(date_column AS DATE)) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) THEN 'PASS' ELSE 'FAIL' END as freshness_test,
  CASE WHEN COUNT(CASE WHEN SAFE_CAST(status_column AS STRING) IN ('CP', 'PY') THEN 1 END) > 0 THEN 'PASS' ELSE 'FAIL' END as status_test
FROM `project.dataset.main_table`
WHERE SAFE_CAST(date_column AS DATE) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
LIMIT 1



business_rules_sql: Business logic validation - HANDLE ARRAYS PROPERLY
SELECT 
  'Business Rule Validation' as validation_type,
  'period_type_validation' as rule_name,
  COUNT(*) as records_tested,
  SUM(CASE WHEN SAFE_CAST(period_column AS STRING) IN ('CP', 'PY') THEN 1 ELSE 0 END) as records_passing_rule,
  SAFE_DIVIDE(SUM(CASE WHEN SAFE_CAST(period_column AS STRING) IN ('CP', 'PY') THEN 1 ELSE 0 END), COUNT(*)) * 100 as pass_rate_percentage
FROM `project.dataset.main_table`
WHERE SAFE_CAST(date_column AS DATE) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)



sample_data_sql: Representative sample - CAST DATE COLUMNS
SELECT 
  -- Cast all potentially problematic columns
  SAFE_CAST(date_column AS DATE) as date_column,
  customer_dimension,
  product_dimension,
  SAFE_CAST(fiscal_week_id AS STRING) as fiscal_week_id,
  SAFE_CAST(revenue_metric AS NUMERIC) as revenue_metric,
  SAFE_CAST(volume_metric AS NUMERIC) as volume_metric
FROM `project.dataset.main_table`
WHERE SAFE_CAST(date_column AS DATE) >= DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY)
  AND revenue_metric IS NOT NULL
ORDER BY SAFE_CAST(date_column AS DATE) DESC, SAFE_CAST(revenue_metric AS NUMERIC) DESC
LIMIT 500


BIGQUERY SYNTAX ENFORCEMENT:

Type Casting Rules:

ALWAYS use SAFE_CAST() instead of implicit conversion
Cast before any comparison: SAFE_CAST(column AS STRING) = 'value'
Cast before string functions: LPAD(CAST(number AS STRING), 2, '0')



Aggregation Rules:

Every non-aggregated column in SELECT must be in GROUP BY
Use SUM, COUNT, AVG, MAX, MIN for calculated fields
When unsure, aggregate the column: SUM(column) or MAX(column)



Array Handling:

Never include NULL in arrays: ARRAY(SELECT x FROM UNNEST([col1, col2]) AS x WHERE x IS NOT NULL)
Use ARRAY_AGG() for creating arrays from query results



Date/Time Functions:

Use CURRENT_DATE() not CURRENT_DATE
Use DATE_SUB(CURRENT_DATE(), INTERVAL n DAY)
Cast date columns: SAFE_CAST(date_col AS DATE)



String Comparisons:

Always cast numbers to strings before string comparison
Use SAFE_CAST to avoid errors: SAFE_CAST(fiscal_week AS STRING) = 'CP'


IMPORTANT NOTES:
primary_analysis_sql is the main SQL to execute - this gives you dashboard metrics with real data
All SQL must be valid BigQuery syntax - test every comparison and function
Use SAFE_CAST extensively - prevents type mismatch errors
Always GROUP BY non-aggregated columns - prevents grouping errors
Handle NULLs in arrays explicitly - prevents array errors
Focus on business logic while ensuring technical correctness
Prioritise metrics marked as is_kpi=true and business_criticality=high
All SQL queries should use appropriate sampling and be cost-optimised
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

# Data Processing Utilities
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
                "total_processed_bytes": row['totalProcessedBytes'],
                "sql_query": row['query_text']
            })
        
        # Format the structured prompt for this dashboard
        formatted_prompt = LOOKER_ANALYSIS_PROMPT.format(
            dashboard_id=dashboard_data["dashboard_id"],
            dashboard_name=dashboard_data["dashboard_name"],
            sql_samples=json.dumps(dashboard_data["sql_samples"], indent=2)
        )
        
        batch_data.append({"content": formatted_prompt})
    
    return batch_data

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
