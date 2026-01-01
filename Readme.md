import dagshub
dagshub.init(repo_owner='happinesswhat31', repo_name='MLOPSProject', mlflow=True)

import mlflow
with mlflow.start_run():
mlflow.log_param('parameter name', 'value')
mlflow.log_metric('metric name', 1)

Develped After vscode deployment through colab
