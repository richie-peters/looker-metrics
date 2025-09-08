import json
import re
import os
import sys
import time
from datetime import datetime
import pandas as pd
import subprocess
from google.cloud import aiplatform_v1beta1, bigquery, storage

# ==============================================================================
# 1. MAIN PIPELINE ORCHESTRATOR
# ==============================================================================

def run_pipeline(pipeline_config_path, initial_query_df, session_config, config):
    """
    Runs the entire multi-stage analysis pipeline based on a configuration file.
    """
    try:
        with open(pipeline_config_path, 'r') as f:
            pipeline_config = json.load(f)
    except FileNotFoundError:
        print(f"✗ ERROR: Pipeline configuration file not found at '{pipeline_config_path}'")
        return

    stage_outputs = {"initial_sql_query": initial_query_df}

    for stage in pipeline_config.get("pipeline", []):
        stage_name = stage["stage_name"]
        print(f"\n--- EXECUTING PIPELINE STAGE: {stage_name.upper()} ---")

        input_source_key = stage["input_source"]
        input_data = stage_outputs.get(input_source_key)

        if input_data is None or input_data.empty:
            print(f"⚠️ Skipping stage '{stage_name}' due to missing or empty input data.")
            continue
        
        output_df = _run_analysis_stage(stage, input_data, session_config, config)

        if output_df is not None and not output_df.empty:
            for output_name, output_path in stage["output_files"].items():
                key = f"{stage_name}.{output_name}"
                if len(stage["output_files"]) == 1:
                    stage_outputs[key] = output_df
                else:
                    if output_name == 'dashboards':
                        stage_outputs[key] = output_df[['dashboard_id', 'dashboard_name', 'business_domain']].drop_duplicates()
                    elif output_name == 'metrics':
                        stage_outputs[key] = output_df.drop(columns=['dashboard_name', 'business_domain'])

    print("\n\n✅ Pipeline execution complete.")

# ==============================================================================
# 2. GENERIC STAGE RUNNER
# ==============================================================================

def run_analysis_stage(stage_config, input_data, session_config, config):
    """
    A generic function to run a single stage of the AI analysis pipeline.
    """
    stage_name = stage_config["stage_name"]
    prompt_template = _load_prompt(stage_config["prompt_file"])
    if not prompt_template: return None

    batch_data = _prepare_batch_input(stage_name, input_data, prompt_template, stage_config["group_by"])
    if not batch_data:
        print(f"✗ No data to process for stage: {stage_name}.")
        return None

    gcs_paths = config.GCS_PATHS.get(stage_name)
    if not gcs_paths:
        print(f"✗ ERROR: GCS paths not defined for stage '{stage_name}' in config.py.")
        return None

    results = _run_gemini_batch(
        requests=batch_data,
        project=session_config['project_config']['vertex_project'],
        display_name=f"looker_metric_analysis_{stage_name}",
        input_gcs_uri=gcs_paths["input"],
        output_gcs_uri=gcs_paths["output"],
        model_name=config.GEMINI_MODEL_NAME
    )
    if not results: return None

    # --- MODIFICATION HERE ---
    # The parsing function now returns two dataframes; we capture both.
    main_df, analysis_df = _parse_batch_results(stage_name, results)

    # Pass both dataframes to the saving function.
    _save_stage_output(stage_name, main_df, analysis_df, stage_config["output_files"])
    
    # The main dataframe (metrics) is returned to the pipeline orchestrator.
    return main_df

# ==============================================================================
# 3. HELPER FUNCTIONS (PREPARATION, PARSING, SAVING)
# ==============================================================================

def _load_prompt(prompt_path):
    try:
        with open(prompt_path, "r") as f: return f.read()
    except FileNotFoundError:
        print(f"✗ ERROR: Prompt file not found at '{prompt_path}'")
        return None

def _prepare_batch_input(stage_name, data, prompt_template, group_by_col):
    requests = []
    
    # --- THIS IS THE FIX ---
    # For Stages 2 and 3, we enrich the input data with the necessary context from previous stages.
    if stage_name == 'stage_2_consolidation':
        dashboards_df = pd.read_csv('./data/looker_analysis_dashboards.csv')
        data = pd.merge(data, dashboards_df[['dashboard_id', 'dashboard_name']], on='dashboard_id', how='left')
    elif stage_name == 'stage_3_standardization':
        dashboards_df = pd.read_csv('./data/looker_analysis_dashboards.csv')
        data = pd.merge(data, dashboards_df[['dashboard_id', 'business_domain']], on='dashboard_id', how='left')
    # ---------------------

    for group_key, group in data.groupby(group_by_col):
        format_dict = {}
        if stage_name == 'stage_1_identification':
            format_dict = {"dashboard_id": group_key, "dashboard_name": group.iloc[0]['looker_studio_report_name'], "sql_samples": json.dumps(group[['jobId', 'username', 'query_text']].to_dict('records'), indent=2, default=str)}
        elif stage_name == 'stage_2_consolidation':
            format_dict = {"dashboard_id": group_key, "dashboard_name": group.iloc[0]['dashboard_name'], "all_metrics_json": json.dumps(group[['metric_id', 'metric_name', 'metric_sql_core']].to_dict('records'), indent=2)}
        elif stage_name == 'stage_3_standardization':
            format_dict = {"business_domain": group_key, "all_metrics_json": json.dumps(group[['consolidated_metric_id', 'metric_name', 'business_description', 'consolidated_sql_core']].to_dict('records'), indent=2)}
        
        requests.append({"content": prompt_template.format(**format_dict)})
    return requests

def _parse_batch_results(stage_name, results):
    parsed_items = []
    # This list is new, to hold the dashboard-level analysis from Stage 2
    dashboard_analysis_items = [] 

    for result in results:
        try:
            raw_text = result.get('response', {}).get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            json_text_match = re.search(r'```json\s*(.*?)\s*```', raw_text, re.DOTALL)
            if not json_text_match:
                json_text_match = re.search(r'(\{.*?\})', raw_text, re.DOTALL)
            
            if not json_text_match:
                print("⚠️ Warning: Could not find JSON in a result, skipping.")
                continue

            # Using strict=False can help with minor formatting issues
            parsed_json = json.loads(json_text_match.group(1), strict=False)

            if stage_name == 'stage_1_identification':
                summary = parsed_json.get('dashboard_summary', {})
                if summary.get('dashboard_id') == 'GENERATION_ERROR':
                    continue
                for metric in parsed_json.get('metrics', []):
                    metric.update(summary)
                    parsed_items.append(metric)
            
            # --- THIS IS THE NEW, CORRECTED LOGIC FOR STAGE 2 ---
            elif stage_name == 'stage_2_consolidation':
                analysis = parsed_json.get('dashboard_analysis_and_consolidation', {})
                
                # Extract the dashboard summary and save it separately
                summary = analysis.get('dashboard_summary', {})
                if summary:
                    dashboard_analysis_items.append(summary)

                # Extract the consolidated metrics
                for metric in analysis.get('consolidated_metrics', []):
                    # Add dashboard_id for context, as it's in the summary
                    metric['dashboard_id'] = summary.get('dashboard_id')
                    parsed_items.append(metric)
            # ---------------------------------------------------------

            elif stage_name == 'stage_3_standardization':
                recs = parsed_json.get('domain_metric_recommendations', {})
                for metric in recs.get('recommended_metrics', []):
                    metric['business_domain'] = recs.get('business_domain')
                    parsed_items.append(metric)

        except (json.JSONDecodeError, Exception) as e:
            print(f"✗ ERROR parsing a result: {e}. Skipping.")
            continue
    
    # In Stage 2, we need to handle two different dataframes.
    # For simplicity, this function will now return a tuple: (main_df, optional_analysis_df)
    if stage_name == 'stage_2_consolidation':
        return pd.DataFrame(parsed_items), pd.DataFrame(dashboard_analysis_items)

    return pd.DataFrame(parsed_items), None

def _save_stage_output(stage_name, main_df, analysis_df, output_files_config):
    """
    Saves the output dataframe(s) of a pipeline stage to the specified files.
    For Stage 2, it handles saving both the main metrics and the analysis summary.
    """
    if main_df is None or main_df.empty:
        print(f"⚠️ No main data to save for stage '{stage_name}'.")
        return

    # --- NEW LOGIC FOR STAGE 2 ---
    if stage_name == 'stage_2_consolidation':
        # Save the consolidated metrics dataframe
        metrics_output_path = output_files_config.get("consolidated_metrics")
        if metrics_output_path:
            os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
            main_df.to_csv(metrics_output_path, index=False)
            print(f"✓ Saved output 'consolidated_metrics' to: {metrics_output_path}")

        # Save the separate dashboard analysis dataframe
        analysis_output_path = output_files_config.get("dashboard_analysis")
        if analysis_output_path and analysis_df is not None and not analysis_df.empty:
            os.makedirs(os.path.dirname(analysis_output_path), exist_ok=True)
            analysis_df.to_csv(analysis_output_path, index=False)
            print(f"✓ Saved output 'dashboard_analysis' to: {analysis_output_path}")
        return
    # ----------------------------

    # Original logic for other stages
    if main_df.empty: return
    for output_name, output_path in output_files_config.items():
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if stage_name == 'stage_1_identification' and output_name == 'dashboards':
            main_df[['dashboard_id', 'dashboard_name', 'business_domain']].drop_duplicates().to_csv(output_path, index=False)
        elif stage_name == 'stage_1_identification' and output_name == 'metrics':
            main_df.drop(columns=['dashboard_name', 'business_domain']).to_csv(output_path, index=False)
        else:
            main_df.to_csv(output_path, index=False)
        print(f"✓ Saved output '{output_name}' to: {output_path}")
# ==============================================================================
# 4. CORE UTILITIES (BigQuery, GCS, Git)
# ==============================================================================
def _run_gemini_batch(requests, project, display_name, input_gcs_uri, output_gcs_uri, model_name, max_output_tokens=8192, temperature=0.2):
    print("📤 Preparing batch input...")
    try:
        client = storage.Client()
        bucket_name, blob_path = input_gcs_uri.replace('gs://', '').split('/', 1)
        blob = client.bucket(bucket_name).blob(blob_path)
        jsonl_content = "\n".join([json.dumps({"request": {"contents": [{"role": "user", "parts": [{"text": r["content"]}]}], "generation_config": {"temperature": temperature, "max_output_tokens": max_output_tokens}}}) for r in requests])
        blob.upload_from_string(jsonl_content, content_type='application/jsonl')
    except Exception as e:
        print(f"✗ Failed to prepare batch input: {e}")
        return None

    print("🚀 Creating batch prediction job...")
    job = None
    try:
        api_endpoint = "us-central1-aiplatform.googleapis.com"
        client = aiplatform_v1beta1.JobServiceClient(client_options={"api_endpoint": api_endpoint})
        parent = f"projects/{project}/locations/us-central1"
        job = client.create_batch_prediction_job(parent=parent, batch_prediction_job={"display_name": display_name, "model": f"publishers/google/models/{model_name}", "input_config": {"instances_format": "jsonl", "gcs_source": {"uris": [input_gcs_uri]}}, "output_config": {"predictions_format": "jsonl", "gcs_destination": {"output_uri_prefix": output_gcs_uri}}})
    except Exception as e:
        print(f"✗ Batch prediction job creation failed: {e}")
        return None

    print(f"Monitoring batch job: {job.name.split('/')[-1]}")
    start_time = time.time()
    while True:
        job_state = client.get_batch_prediction_job(name=job.name)
        state_name = job_state.state.name
        elapsed = int(time.time() - start_time)
        sys.stdout.write(f"\rStatus: {state_name} | Elapsed: {elapsed//60}m {elapsed%60}s")
        sys.stdout.flush()
        if "SUCCEEDED" in state_name or "FAILED" in state_name or "CANCELLED" in state_name:
            sys.stdout.write("\n")
            if "SUCCEEDED" in state_name:
                print("📊 Reading results from the latest job...")
                try:
                    exact_output_uri = job_state.output_info.gcs_output_directory
                    bucket_name, prefix = exact_output_uri.replace('gs://', '').split('/', 1)
                    blobs = storage.Client().list_blobs(bucket_name, prefix=prefix)
                    results = []
                    for blob in blobs:
                        if 'predictions' in blob.name and blob.name.endswith('.jsonl'):
                            for line in blob.download_as_text().splitlines():
                                if line.strip():
                                    results.append(json.loads(line))
                    print(f"✓ Successfully read {len(results)} predictions.")
                    return results
                except Exception as e:
                    print(f"✗ Failed to read batch results: {e}")
            else:
                print(f"❌ Batch job finished with status: {state_name}")
            return None
        time.sleep(10)

def run_sql_file(file_path, replacements=None, project=None):
    try:
        with open(file_path, 'r', encoding='utf-8') as f: sql = f.read()
        if replacements:
            for old, new in replacements.items(): sql = sql.replace(old, new)
        client = bigquery.Client(project=project)
        df = client.query(sql).to_dataframe()
        print(f"✓ Query executed successfully - returned {len(df)} rows")
        return df
    except Exception as e:
        print(f"✗ Failed to run SQL file {file_path}: {e}")
        return None


# ==============================================================================
# 5. GIT UTILITY FUNCTIONS
# ==============================================================================

def fix_github_authentication():
    """
    Prompts for a GitHub Personal Access Token and securely updates the remote URL.
    """
    TOKEN = input("Paste your GitHub Personal Access Token here: ")
    if not TOKEN or TOKEN.strip() == "":
        print("❌ No token provided!")
        return False
    
    # It's safer to use a specific username, but this can be adapted if needed
    USERNAME = "richie-peters"
    TOKEN = TOKEN.strip()
    repo_url = f"https://{USERNAME}:{TOKEN}@github.com/richie-peters/looker-metrics.git"
    
    try:
        # Assumes the notebook is running inside the cloned directory
        cwd = '.' 
        result = subprocess.run(
            f'git remote set-url origin {repo_url}',
            shell=True, capture_output=True, text=True, check=True, cwd=cwd
        )
        print("✅ GitHub authentication updated successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to set remote URL: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred during authentication setup: {e}")
        return False

def auto_git_push(commit_message="Auto-commit from Colab", files_to_add="."):
    """
    Automatically adds, commits (with a timestamp), and pushes changes to GitHub.
    """
    print("🔄 Auto Git Push Starting...")
    cwd = '.' # Assumes the notebook is running inside the cloned directory

    def run_git(command):
        return subprocess.run(command, shell=True, capture_output=True, text=True, cwd=cwd)

    # Check for changes
    status_result = run_git("git status --porcelain")
    if not status_result.stdout.strip():
        print("✅ No changes to commit.")
        return True

    # Add, commit, and push
    print(f"   - Adding files: {files_to_add}")
    run_git(f"git add {files_to_add}")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final_commit_message = f'"{commit_message} - {timestamp}"'
    print(f"   - Committing with message: {final_commit_message}")
    commit_result = run_git(f'git commit -m {final_commit_message}')

    if commit_result.returncode != 0 and "nothing to commit" not in commit_result.stdout:
        print(f"❌ Failed to commit: {commit_result.stderr}")
        return False

    print("   - Pushing to remote 'origin'...")
    push_result = run_git("git push origin main")

    if push_result.returncode != 0:
        print(f"❌ Failed to push: {push_result.stderr}")
        return False

    print("✅ Successfully pushed to GitHub!")
    return True