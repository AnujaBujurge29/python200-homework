from prefect import task, flow
import numpy as np
import pandas as pd


@task
def create_series(arr):
    return pd.Series(arr, name="values")


@task
def clean_data(series):
    return series.dropna()


@task
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }


@flow
def pipeline_flow():
    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan,
                    18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

    series = create_series(arr)
    clean = clean_data(series)
    summary = summarize_data(clean)

    for key, value in summary.items():
        print(f"{key}: {value}")

    return summary


if __name__ == "__main__":
    pipeline_flow()


# Question 1:
# This pipeline is simple -- just three small functions on a handful of numbers. Why might Prefect be more overhead than it is worth here?
# Answer:
# Prefect adds overhead here because the pipeline is small, fast, and has no external dependancies
# or failure risks

# Question 2:
# Describe some realistic scenarios where a framework like Prefect could still be useful, even if the pipeline logic itself stays simple like in this case.
# Answer:
# Prefect becomes useful when pipeline includes:
# - long-running job
# - error handling
# - scheduling
# - logging and monitoring
# - parallel execution
