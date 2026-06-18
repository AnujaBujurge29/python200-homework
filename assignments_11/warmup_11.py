import os
from prefect import get_run_logger

os.system('cls')

# =====================================================================================
#                    Week 11 - Part 1: Warmup Exercise
# =====================================================================================

print('='*100)
print("Week 11 - Part 1: Warmup Exercise")
print('='*100)

# =====================================================================================
# Prefect Orchestration
# =====================================================================================

print("Prefect Orchestration")

# =====================================================================================
# Prefect Orchestration - Question 1
# =====================================================================================

print('-'*100)
print("Prefect Orchestration - Question 1")
print('-'*100)

# Question:
# what is the difference between a @task and a @flow in Prefect?
# Answer
# A @task in Prefect is a unit of work that can be retried, cached, and executed independently.
# A @flow is the main orchestration function that defines how tasks are composed and run.
#
# For a pure, in-memory helper function that only converts Celsius to Fahrenheit and has no I/O,
# I would not normally decorate it with @task. Making it a plain function keeps it simple and
# avoids extra orchestration overhead. Use @task when you want Prefect-specific features like
# retries, caching, logging, or parallel execution.

print("Comment added for Prefect Orchestration - Question 1")

# =====================================================================================
# Prefect Orchestration - Question 2
# =====================================================================================

print('-'*100)
print("Prefect Orchestration - Question 2")
print('-'*100)

# @task(retries=3, retry_delay_seconds=30)

print("Comment added for Prefect Orchestration - Question 2")

# =====================================================================================
# Prefect Orchestration - Question 3
# =====================================================================================

print('-'*100)
print("Prefect Orchestration - Question 3")
print('-'*100)

# Question:
# where in the UI do you look to understand what went wrong, and what specific information
# would you expect to find there?
# Answer:
# In the Prefect UI I would open the flow run details for the pipeline execution, then
# inspect the "Task Runs" or "Flow Run" graph/summary. I would look for the failed
# transform task and click into its log/output details.
#
# I expect to find the error message or exception stack trace for transform, the timestamp
# when it failed, and any retry history. This information tells me why transform failed and
# whether the load step was skipped because the flow stopped or because transform was
# configured as a dependency for load.

print("Comment added for Prefect Orchestration - Question 3")

# =====================================================================================
# Production Patterns
# =====================================================================================

print("Production Patterns")

# =====================================================================================
# Production Patterns - Question 1
# =====================================================================================

print('-'*100)
print("Production Patterns - Question 1")
print('-'*100)

# Question:
# What does raise_for_status() do and why is it better than writing
# if response.status_code != 200: print("error") in a pipeline task?
# What happens to downstream tasks in each case when the API returns a 500 error?
# Answer:
# raise_for_status() checks the HTTP response status code and raises an exception
# when the response indicates a client or server error (4xx or 5xx). This makes the
# failure explicit and allows the pipeline orchestration system to detect the error.
#
# In a pipeline task, raise_for_status() is better because it fails fast and triggers
# proper error handling, retries, and task-state propagation.
#
# If you only print("error") when status_code != 200, the task may appear to have
# succeeded even though the API call failed. Downstream tasks may continue running on
# invalid or missing data, causing harder-to-debug failures later.
#
# When the API returns a 500 error, raise_for_status() will raise an HTTPError and
# the current task will fail, preventing downstream tasks from running unless retries
# or recovery logic are configured. If you just print("error"), the task may still
# return normally and Prefect could continue to execute downstream tasks, leading to
# incorrect results or cascading failures.

print("Comment added for Production Patterns - Question 1")

# =====================================================================================
# Production Patterns - Question 2
# =====================================================================================

print('-'*100)
print("Production Patterns - Question 2")
print('-'*100)

# Question:
# Your pipeline uploads results to final/{today}/weather_etl.json with overwrite=True.
# The pipeline crashes halfway through the transform step. You fix the bug and re-run it
# from the beginning. What does overwrite=True protect you from in this scenario, and
# what would happen without it?
# Answer:
# overwrite=True protects you from stale or partially written output remaining from the
# previous failed run. When the pipeline re-runs, it replaces the old file with the new
# correct output, ensuring consumers only see the complete, updated data.
#
# Without overwrite=True, the upload could fail if the file already exists, or the
# destination might retain partial or corrupt data from the prior run. This can lead to
# inconsistent downstream reads, because the file may contain a mix of old and new data,
# or the pipeline may error out on a second attempt when the existing file blocks upload.

print("Comment added for Production Patterns - Question 1")

# =====================================================================================
# Production Patterns - Question 3
# =====================================================================================

print('-'*100)
print("Production Patterns - Question 3")
print('-'*100)

# @task
# def log_load_summary(records: list, blob_path: str):
#     logger = get_run_logger()
#     logger.info(f"Loaded {len(records)} records into {blob_path}")

print("Production Patterns")

print("Comment added for Production Patterns - Question 1")
