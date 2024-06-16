import os
import json
def save_experiment_results(filename, variables, results):
    os.makedirs('MSE_logs', exist_ok=True)
    filepath = os.path.join('MSE_logs', filename)
    data = {
        'variables': variables,
        'results': results
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_experiment_results(filename):
    filepath = os.path.join('MSE_logs', filename)
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['variables'], data['results']
