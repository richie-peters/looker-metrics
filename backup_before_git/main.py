"""
Looker Studio Analysis Pipeline - Direct Execution
==================================================
"""

import pandas as pd
import config
import functions

# Configuration
EMAIL_LIST = ["richie.peters@news.com.au"]
global_results = None

# 1. Initialise session
session_config = config.initialize_session()
config.print_session_summary(session_config)

# 2. Extract data from BigQuery
print("\n--- Extracting data from BigQuery ---")
replacements = {'INSERT': functions.format_emails_for_sql(EMAIL_LIST)}
query_df = functions.run_sql_file(config.LOOKER_SQL_FILE, replacements=replacements, project=config.BQ_PROJECT_ID)

if query_df is None or query_df.empty:
    print("✗ No data extracted from BigQuery. Exiting.")
else:
    # 3. Prepare batch input for Gemini
    print("\n--- Preparing batch input for Gemini ---")
    batch_data = functions.prepare_looker_analysis_batch(query_df)

    # 4. Run Gemini batch prediction (SLICK VERSION)
    print("\n--- Running Gemini batch prediction ---")
    results = functions.run_gemini_batch_fast_slick(
        requests=batch_data,
        project=session_config['project_config']['vertex_project'],
        display_name="looker-analysis-batch",
        input_gcs_uri=config.INPUT_GCS_URI,
        output_gcs_uri=config.OUTPUT_GCS_URI,
        model_name=config.GEMINI_MODEL_NAME
    )

    if results is None:
        print("✗ Gemini batch prediction failed. Exiting.")
    else:
        # Make results global for troubleshooting
        global_results = results
        print(f"✓ Results stored in global_results variable ({len(results)} items)")

        # 5. Process and save results
        print("\n--- Processing results ---")
        datasets = functions.convert_batch_results_to_dataset(results)
        if datasets:
            functions.save_datasets_to_csv(datasets, "./data/")
            functions.analyse_results_summary(datasets)

        print(f"✓ Successfully processed {len(results)} results.")