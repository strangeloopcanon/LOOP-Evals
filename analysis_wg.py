import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Ensure output directories exist
os.makedirs('charts', exist_ok=True)
archive_folder_path = '#Archive/'
os.makedirs(archive_folder_path, exist_ok=True)

combined_results_path = 'results/results_wg.json'  # Path to your multi-model JSON

with open(combined_results_path, 'r') as file:
    data = json.load(file)
    # data now looks like:
    # {
    #   "chatgpt-4o-latest": {
    #       "matrix_objective_4": [... attempts ...],
    #       "matrix_objective_5": [... attempts ...]
    #    },
    #   "claude-3-5-sonnet-latest": {
    #       "matrix_objective_4": [... attempts ...],
    #       "matrix_objective_5": [...]
    #    },
    #    ...
    # }

def compute_metrics_for_attempts(attempts_list):
    """
    Each item in attempts_list is typically something like:
      {
        'attempt_number': 1,
        'llm_type': 'some-model',
        'runs': [
           { 'index':1, 'matrix': [...], 'word_responses': [...], 'false_count': 2, 'error': None },
           { 'index':2, 'matrix': [...], 'false_count': 1, 'error': None },
           ...
        ],
        'success': True/False
      }
    We want to compute overall success rate, average false_count, etc.
    """
    total_attempts = len(attempts_list)
    successes = 0
    total_runs = 0
    total_false_count = 0
    all_false_counts = []

    for attempt in attempts_list:
        if attempt.get('success'):
            successes += 1

        if 'runs' in attempt and attempt['runs']:
            for run in attempt['runs']:
                fc = run.get('false_count', 0)
                total_false_count += fc
                all_false_counts.append(fc)
                total_runs += 1

    success_rate = successes / total_attempts if total_attempts else 0
    avg_false_count = total_false_count / total_runs if total_runs else 0

    return success_rate, avg_false_count, all_false_counts

# We'll gather a table of (model, matrix_name, success_rate, avg_false_count, all_false_counts)
records = []

for model_name, model_dict in data.items():
    # model_dict is like { "matrix_objective_4": [ ... attempts ...], "matrix_objective_5": [... attempts ...] }
    for matrix_name, attempts_list in model_dict.items():
        sr, afc, afc_list = compute_metrics_for_attempts(attempts_list)
        records.append({
            'Model': model_name,
            'Matrix': matrix_name,
            'SuccessRate': sr,
            'AvgFalseCount': afc,
            'AllFalseCounts': afc_list
        })

# Convert to a DataFrame
df = pd.DataFrame(records)

# ============== Plot #1: Success Rate by (Model, Matrix) ==============
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Matrix', y='SuccessRate', hue='Model', alpha=0.7)
plt.title('Success Rate by Model and Matrix')
plt.ylabel('Success Rate')
plt.ylim(0, 1)
plt.legend(title='Model')
plt.savefig('charts/wg_success_rate_by_model.png')
plt.show()

# ============== Plot #2: Average False Count by (Model, Matrix) ==============
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Matrix', y='AvgFalseCount', hue='Model', alpha=0.7)
plt.title('Average False Count by Model and Matrix')
plt.ylabel('Avg False Count')
plt.legend(title='Model')
plt.savefig('charts/wg_avg_false_count_by_model.png')
plt.show()

# ============== Plot #3: Distribution of false counts (hist) ==============
# We might want a separate distribution per (Model, Matrix)
# We'll explode these rows so that each false count is a separate row in the DF
dist_rows = []
for _, row in df.iterrows():
    for fc in row['AllFalseCounts']:
        dist_rows.append({
            'Model': row['Model'],
            'Matrix': row['Matrix'],
            'FalseCount': fc
        })
dist_df = pd.DataFrame(dist_rows)

plt.figure(figsize=(10, 6))
sns.histplot(
    data=dist_df, 
    x='FalseCount', 
    hue='Matrix', 
    multiple='stack', 
    bins=range(0, dist_df['FalseCount'].max() + 2), 
    alpha=0.7
)
plt.title("Distribution of False Counts (stacked by Matrix)")
plt.savefig('charts/wg_falsecount_distribution.png')
plt.show()

print("Analysis done. See charts folder for the new plots.")
