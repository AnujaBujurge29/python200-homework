from smolagents import ToolCallingAgent, OpenAIServerModel, tool, CodeAgent
import os
from dotenv import load_dotenv
from pathlib import Path
from scipy import stats
import pandas as pd
import matplotlib
matplotlib.use("Agg")

# smolagents imports

os.system('cls')

# =====================================================================================
#                    Week 7 - Part 1: Mini-Project — World Happiness Agent
# =====================================================================================

print('='*100)
print(" Part 1: Mini-Project — World Happiness Agent")
print('='*100)

# =====================================================================================
# Pre-task: Load the Data
# =====================================================================================

# print('='*100)
print("---------------------- Pre-task: Load the Data ----------------------")
print('='*100)

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# Global dataframe
DATA_PATH = "../assignments_01/outputs/merged_happiness.csv"
df = None

# =====================================================================================
# Task 1: Define Your Tools
# =====================================================================================

print('='*100)
print(" Task 1: Define Your Tools")
print('='*100)


@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.
    ...
    Loads the merged CSV from DATA_PATH. If that file does not exist,
    falls back to loading and merging all yearly CSVs from the happiness
    project resources folder.

    Return:
        A dict with "shape" (tuple of rows and columns) and "columns"
        (list of column names), or an error dict if loading fails.
    """

    global df

    path = Path(DATA_PATH)
    try:
        if path.exists():
            df = pd.read_csv(path)
        else:
            resources_dir = Path(
                "../../python-200/assignments/resources/happiness_project")
            if not resources_dir.exists():
                return {"error": f"Neither '{DATA_PATH}' nor fallback directory found."}
            csv_files = sorted(resources_dir.glob("*.csv"))
            if not csv_files:
                return {"error": "No CSV files found in fallback directory"}
            frames = []
            for csv_file in csv_files:
                frames.append(pd.read_csv(csv_file))
            df = pd.concat(frames, ignore_index=True)
    except Exception as err:
        return {"error": f"Failed to load CSV data: {err}"}

    return {
        "shape": df.shape,
        "columns": df.columns.tolist()
    }


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.
    Uses pandas describe() to compute count, mean, std, min, percentiles,
    and max for numeric columns or count, unique, top, freq for categorical.

    Args:
        column: The name of the column to summarize.

    Returns:
        A dict of descriptive statistics for the column, or an error dict
        if no data is loaded or the column is not found.
    """

    global df

    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}

    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {df.columns.tolist()}"}

    summary = df[column].describe().to_dict()
    # Round numeric values for readability
    cleaned = {}
    for key, value in summary.items():
        if isinstance(value, float):
            cleaned[key] = round(value, 4)
        else:
            cleaned[key] = value
    return cleaned


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Uses scipy.stats.pearsonr to measure the linear relationship between
    two columns in the loaded World Happiness dataset.

    Args:
        col1: The name of the first numeric column.
        col2: The name of the second numeric column.

    Returns:
        A dict with "col1", "col2", "pearson_r", and "p_value" (each float
        rounded to 4 decimal places), or an error dict if input is invalid.
    """
    global df

    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}

    missing = [c for c in [col1, col2] if c not in df.columns]
    if missing:
        return {"error": f"Column(s) not found: {missing}. Available: {df.columns.tolist()}"}

    # Drop rows with NaN in either column for a clean correlation
    clean = df[[col1, col2]].dropna()
    if len(clean) < 3:
        return {"error": f"Not enough non-null data to compute correlation (only {len(clean)} rows)."}

    pearson_r, p_value = stats.pearsonr(clean[col1], clean[col2])

    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(pearson_r, 4),
        "p_value": round(p_value, 4),
    }


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Filters the dataset to the specified year, sorts by the given column in
    descending order, and returns the top N rows.

    Args:
        column: The column to rank countries by (e.g. 'Ladder score').
        year: The year to filter the data to (e.g. 2020).
        n: The number of top countries to return. Defaults to 5.

    Returns:
        A dict with "year", "column", "top_n" (a list of dicts, each with
        "country" and the column value), or an error dict if input is invalid.
    """
    global df

    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}

    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {df.columns.tolist()}"}

    if "Year" not in df.columns:
        return {"error": "'Year' column not found in dataset."}

    year_df = df[df["Year"] == year]
    if year_df.empty:
        available_years = sorted(df["Year"].unique().tolist())
        return {"error": f"No data for year {year}. Available years: {available_years}"}

    sorted_df = year_df.sort_values(by=column, ascending=False).head(n)

    top_n = []
    for _, row in sorted_df.iterrows():
        top_n.append({
            "country": row["Country"],
            column: round(row[column], 4) if isinstance(row[column], float) else row[column],
        })

    return {
        "year": year,
        "column": column,
        "top_n": top_n,
    }


print("All 4 tools defined: load_happiness_data, summarize_column, compute_correlation, get_top_n_countries")

# =====================================================================================
# Task 2: Build the Agent
# =====================================================================================

print('='*100)
print(" Task 2: Build the Agent")
print('='*100)

api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column,
           compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=[
        "pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)

print("Code Agent Created.")

# =====================================================================================
# Task 3: Run Guided Queries
# =====================================================================================

print('='*100)
print(" Task 3: Run Guided Queries")
print('='*100)

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)

    if os.path.exists("outputs/happiness_by_region.png"):
        print("\nPlot saved successfully")
    else:
        print("\nWarning: Plot files was not saved")

    # =====================================================================================
    # Task 4: Your Own Questions
    # =====================================================================================

    print('='*100)
    print(" Task 4: Your Own Questions")
    print('='*100)

    # My query 1
    # replace with your question
    my_query_1 = "Which country had the biggest improvement in Ladder score between 2015 and 2020"
    response_1 = agent.run(my_query_1, reset=False)
    print("\nQuery 1:")
    print(response_1)
    # Comment: Did this trigger tool use, code generation, or both?
    # this should trigger code generation since no tool computes differneces
    # across years. the agent needs to filter, merge/pivot, and compute deltas - all
    # requirening custom pandas.

    # My query 2
    # replace with your question
    my_query_2 = "What is the correlation between Freedom to make life chices and Ladder Score"
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)
    # Comment: Did this trigger tool use, code generation, or both?
    # This should trigger tool use (compute_correlation) since
    # it's a direct Pearson Correlation between two columns -
    # exactly what the tool is disigned for.

# =====================================================================================
# Task 5: Reflection
# =====================================================================================

print('='*100)
print(" Task 5: Reflection")
print('='*100)

# =====================================================================================
# Task 5: Reflection
# =====================================================================================

# --- Reflection ---
#
# 1. In Query 3, how did the agent communicate whether the correlation was statistically
#     significant? Did it use the p-value correctly? What threshold did it apply?
#
#     The agent used the compute_correlation tool which returned both the Pearson r value
#     and the p-value. It then interpreted the p-value by comparing it against the standard
#     significance threshold of 0.05 (alpha = 5%). Since the p-value for GDP per capita vs
#     Ladder score is extremely small (effectively 0.0), the agent correctly stated that the
#     correlation is statistically significant — meaning the relationship is very unlikely
#     to have occurred by chance. The agent used the p-value correctly.
#
# 2. Did any of the agent's responses surprise you — either by being more capable than
#     you expected, or less? Describe one specific example.
#
#     Query 5 was impressive: the agent wrote full matplotlib code to create a multi-line
#     plot grouped by region, including a legend, axis labels, and saving to a file — all
#     without any plot-specific tool. It correctly used groupby on the Regional indicator
#     column and iterated over groups to plot each line. The ability to go from a natural
#     language request to working visualization code in one shot was more capable than
#     expected.
#
# 3. What one additional tool would make this agent meaningfully more useful?
#     Describe what it would do and what kind of question it would help the agent answer.
#
#     A "compare_countries" tool that takes two country names and a list of columns, and
#     returns a side-by-side comparison across all available years. This would help answer
#     questions like "How does Finland compare to the US in GDP per capita and social support
#     over time?" without requiring the agent to write custom filtering and reshaping code
#     every time. It would make comparative analysis queries fast, reliable, and less
#     error-prone.

print("Comment added")
