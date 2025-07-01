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
6. **UNIFIED ANALYSIS**: Create consolidated queries that analyze all metrics together
7. **DATA GOVERNANCE**: Identify hardcoded values that should be parameterised or moved to reference tables

OUTPUT REQUIREMENTS (JSON):
Return a flat JSON structure with the following schema:

{{
  "dashboard_summary": {{
    "dashboard_id": "string",
    "dashboard_name": "string",
    "primary_data_sources": ["table1", "table2"],
    "business_domain": "advertising/finance/consumer/operations/etc",
    "complexity_score": 1-10,
    "total_metrics_identified": number,
    "data_grain": "transaction/daily/weekly/monthly/customer/product",
    "key_dimensions": ["date", "customer_id", "product", "region"],
    "date_range_detected": "approximate date range found in data",
    "primary_aggregation_level": "daily/weekly/monthly/yearly",
    "main_business_entities": ["customers", "products", "transactions"]
  }},
  "data_governance_flags": {{
    "has_hardcoded_dates": true/false,
    "hardcoded_dates_found": ["2024-07-03", "2025-03-30", "specific dates found in SQL"],
    "hardcoded_date_contexts": ["WHERE date >= '2024-01-01'", "other examples of hardcoded date usage"],
    "has_hardcoded_business_logic": true/false,
    "hardcoded_business_logic": [
      {{
        "logic_type": "publication_mapping/status_values/category_definitions/etc",
        "hardcoded_values": ["'TA'", "'DT'", "'HS'", "specific hardcoded values"],
        "sql_context": "CASE WHEN publication = 'TA' THEN 'The Australian'",
        "governance_recommendation": "Move to publication_reference_table",
        "impact_assessment": "high/medium/low - how often this logic appears"
      }}
    ],
    "parameterisation_opportunities": [
      {{
        "parameter_type": "date_range/business_rules/filter_values",
        "current_hardcoded_approach": "example of current SQL",
        "recommended_approach": "example of improved SQL with joins/parameters",
        "benefit": "maintainability/consistency/governance"
      }}
    ]
  }},
  "dataset_analysis": {{
    "structure_sql": "Query to understand data structure, grain, and key dimensions with sampling",
    "metrics_combined_sql": "** THIS IS THE MAIN SQL TO EXECUTE ** - Single query showing all key metrics calculated together with appropriate sampling and date filters",
    "validation_sql": "Quick validation that all metric calculations work syntactically", 
    "business_rules_sql": "Query to validate key business logic, filters, and data quality"
  }},
  "metrics": [
    {{
      "metric_id": "unique_identifier",
      "metric_name": "human_readable_name",
      "metric_type": "dimension/measure/calculated_field/filter/aggregation",
      "calculation_type": "sum/count/average/ratio/case_when/date_function/etc",
      "is_final_output": true/false,
      "depends_on_metrics": ["metric_id1", "metric_id2"],
      "business_description": "what this metric represents in business terms",
      "sql_logic": "core SQL calculation logic extracted from queries",
      "data_sources": ["source_table1", "source_table2"],
      "filters_applied": ["date filters", "business rules", "exclusions"],
      "aggregation_level": "transaction/daily/monthly/customer/product",
      "expected_data_type": "numeric/string/date/boolean",
      "business_context": "how this metric fits into overall business analysis",
      "has_hardcoded_values": true/false,
      "hardcoded_values_used": ["specific hardcoded values in this metric calculation"]
    }}
  ],
  "metric_interactions": [
    {{
      "interaction_type": "mathematical_relationship/dependency/filter_impact",
      "primary_metric": "metric_id",
      "related_metrics": ["metric_id1", "metric_id2"],
      "relationship_description": "how these metrics relate to each other",
      "mathematical_formula": "if applicable: A = B * C or A = B + C",
      "business_validation": "what this relationship means for business analysis",
      "potential_data_quality_check": "how to validate this relationship holds true"
    }}
  ]
}}

DATASET ANALYSIS REQUIREMENTS:

1. **structure_sql**: Understand the data grain and key dimensions
   ```sql
   -- Example structure:
   SELECT 
     COUNT(*) as total_records,
     COUNT(DISTINCT date_column) as unique_dates,
     COUNT(DISTINCT customer_column) as unique_customers,
     COUNT(DISTINCT product_column) as unique_products,
     MIN(date_column) as earliest_date,
     MAX(date_column) as latest_date,
     -- Include other key dimensions found
   FROM main_table 
   WHERE date_column >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)


metrics_combined_sql: ** PRIMARY EXECUTION SQL ** - Calculate ALL key metrics in one query
-- THIS IS THE MAIN SQL THAT WILL BE EXECUTED FOR EACH DASHBOARD
-- Example structure:
SELECT 
  date_dimension,
  primary_grouping_dimension,
  COUNT(*) as record_count,
  SUM(revenue_column) as total_revenue,
  COUNT(DISTINCT customer_column) as unique_customers,
  AVG(revenue_column) as avg_revenue_per_record,
  SUM(revenue_column) / COUNT(DISTINCT customer_column) as revenue_per_customer,
  -- Include all other key metrics from dashboard
FROM main_table_or_view
WHERE date_column >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
  AND other_business_filters
GROUP BY date_dimension, primary_grouping_dimension
ORDER BY date_dimension DESC
LIMIT 100



validation_sql: Quick syntax and logic validation
-- Example structure:
SELECT 
  -- Test each major calculation
  CASE WHEN SUM(revenue) > 0 THEN 'PASS' ELSE 'FAIL' END as revenue_test,
  CASE WHEN COUNT(DISTINCT customer) > 0 THEN 'PASS' ELSE 'FAIL' END as customer_test,
  -- Include other key validations
FROM main_table 
WHERE date_column = (SELECT MAX(date_column) FROM main_table)
LIMIT 1



business_rules_sql: Validate business logic and data quality
-- Example structure:
SELECT 
  business_rule_name,
  COUNT(*) as records_affected,
  SUM(CASE WHEN business_condition THEN 1 ELSE 0 END) as records_passing_rule,
  SAFE_DIVIDE(SUM(CASE WHEN business_condition THEN 1 ELSE 0 END), COUNT(*)) as pass_rate
FROM main_table
WHERE date_column >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
GROUP BY business_rule_name


DATA GOVERNANCE ANALYSIS REQUIREMENTS:

Hardcoded Date Detection: Look for specific dates in SQL

Examples: '2024-07-03', '2025-03-30', DATETIME '2024-07-03T00:00:00'
Context: WHERE clauses, date calculations, fixed date references
Flag any date that isn't CURRENT_DATE(), DATE_SUB(), or similar dynamic functions



Hardcoded Business Logic Detection: Look for business rules that should be in reference tables

Publication codes: 'TA', 'DT', 'HS', 'CM' etc.
Status mappings: 'ACTIVE', 'INACTIVE', business status codes
Category definitions: product categories, customer segments, revenue types
Geographic mappings: state codes, region definitions
Business rules: complex CASE statements with multiple hardcoded values



Parameterisation Opportunities: Identify where joins to reference tables would improve governance

Instead of: CASE WHEN publication = 'TA' THEN 'The Australian' WHEN publication = 'DT' THEN 'Daily Telegraph'
Recommend: JOIN publication_reference_table ON t.publication = ref.publication_code


IMPORTANT NOTES:
PRIMARY EXECUTION FIELD: The metrics_combined_sql field contains the main SQL query that will be executed for each dashboard to get actual data insights
Looker SQL uses generic column aliases (clmn0_, clmn1_) - look for real column names in innermost SELECT statements
Focus on business logic in CASE statements, WHERE clauses, and GROUP BY statements
All dataset_analysis SQL should use appropriate sampling (date filters + LIMIT)
Prioritise understanding data grain (transaction vs daily vs monthly aggregation)
Look for key business dimensions that group the data (customer, product, time period, geography)
Identify which metrics are intermediate calculations vs final dashboard outputs
Pay attention to complex business rules in nested CASE statements
Look for data quality filters and exclusions in WHERE clauses
Understand how different metrics relate mathematically (ratios, sums, averages)
Focus on recent data analysis (last 30-90 days) unless historical analysis is specifically needed
Each SQL query should be optimised for cost and performance while providing meaningful insights
Look for patterns in how Looker Studio constructs queries (common CTEs, repeated logic, etc.)
Data Governance Priority: Flag hardcoded dates and business logic as high priority governance issues
Execution Priority: The metrics_combined_sql should be the primary query for getting actual data insights from each dashboard
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
    """Convert batch prediction results to structured dataset - FIXED"""
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
        
        # Parse JSON responses and flatten
        dashboard_data = []
        metrics_data = []
        
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
                    dashboard_summary = parsed_response['dashboard_summary']
                    dashboard_summary['response_id'] = response['response_id']
                    dashboard_data.append(dashboard_summary)
                
                # Extract metrics
                if 'metrics' in parsed_response:
                    for metric in parsed_response['metrics']:
                        metric['response_id'] = response['response_id']
                        metric['dashboard_id'] = parsed_response.get('dashboard_summary', {}).get('dashboard_id', '')
                        metric['dashboard_name'] = parsed_response.get('dashboard_summary', {}).get('dashboard_name', '')
                        metrics_data.append(metric)
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error for response {response['response_id']}: {e}")
                print(f"Response preview: {response['raw_response'][:200]}...")
            except Exception as e:
                print(f"Error processing response {response['response_id']}: {e}")
        
        # Create DataFrames
        dashboards_df = pd.DataFrame(dashboard_data) if dashboard_data else pd.DataFrame()
        metrics_df = pd.DataFrame(metrics_data) if metrics_data else pd.DataFrame()
        
        print(f"✓ Created datasets:")
        print(f"  - Dashboards: {len(dashboards_df)} rows")
        print(f"  - Metrics: {len(metrics_df)} rows")
        
        return {
            'dashboards': dashboards_df,
            'metrics': metrics_df,
            'raw_responses': pd.DataFrame(responses)
        }
        
    except Exception as e:
        print(f"✗ Failed to convert results: {e}")
        return None

def save_datasets_to_csv(datasets, output_folder="./data/"):
    """Save datasets to CSV files"""
    try:
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


    def auto_git_save():
    """Automatically save code to GitHub"""
    import os
    from datetime import datetime
    
    try:
        # Make sure we're in the git repository
        if not os.path.exists('.git'):
            print("⚠️ Not in git repository - changing to /content/looker-metrics")
            os.chdir('/content/looker-metrics')
            
            # Copy the main.py file to git repo
            import shutil
            shutil.copy('/content/main.py', './main.py')
            shutil.copy('/content/config.py', './config.py')  
            shutil.copy('/content/functions.py', './functions.py')
            print("✓ Copied files to git repository")
        
        # Create commit message with timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        commit_msg = f"Auto-save: Analysis run completed at {timestamp}"
        
        # Git commands
        os.system('git add .')
        os.system(f'git commit -m "{commit_msg}"')
        os.system('git push origin main')
        
        print(f"✓ Auto-saved to GitHub: {commit_msg}")
        
    except Exception as e:
        print(f"⚠️ Auto-save failed (but analysis completed): {str(e)}")


