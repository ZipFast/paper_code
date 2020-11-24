from run_loader import RunLoader
import json
import requests

tasks, _ = RunLoader.get_cc18_benchmarking_suite()
flow_id = 6969
metric = "predictive_accuracy"
for task in tasks:
    heads, values = RunLoader.load_task_runs(task, flow_id, metric)
    print(heads)
    print(values)
