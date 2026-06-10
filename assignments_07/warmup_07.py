import os
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# smolagents imports
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent

os.system('cls')
# =====================================================================================
#                    Week 7 - Part 1: Warm up Exercise
# =====================================================================================

print('='*100)
print(" Part 1: Warmup Exercises")
print('='*100)

# =====================================================================================
# Lesson 02: Tool Definitions and the ReAct Loop
# =====================================================================================

# print('='*100)
print("---------------------- Lesson 02: Tool Definitions and the ReAct Loop ----------------------")
# print('='*100)

# =====================================================================================
# Question 1
# =====================================================================================

print('='*100)
print(" Question 1")
print('='*100)

# Fucntion defination


def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"


# JSON schema dictionary describing the function to an LLM
celsius_to_fahrenheit_schema = {
    "name": "celsius_to_fahrenheit",
    "description": "Convert a Celsius temperature to Fahrenheit and return it as a formatted string.",
    "parameters": {
        "type": "object",
        "properties": {
            "celsius": {
                "type": "number",
                "description": "The temperature in Celsius to convert to Fahrenheit"
            }
        },
        "required": ["celsius"]
    }
}

# Print the schema
print("\nTool Schema:")
print(celsius_to_fahrenheit_schema)

# Call the function with 0, 100, and -40
print("\nFunction calls:")
print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))

# =====================================================================================
# Question 2
# =====================================================================================

print('='*100)
print(" Question 2")
print('='*100)

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

client = OpenAI()


def get_current_time() -> str:
    """Return the current date and time as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_agent(prompt: str) -> str:
    """Run a simple agent that has access to get_current_time as its only tool."""
    tool_schema = {
        "name": "get_current_time",
        "description": "Get the current date and time when asked.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that should only use the get_current_time tool when the user explicitly asks for the current date or time. "
                "If the user asks anything else, answer directly without invoking the tool."
            )
        },
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        functions=[tool_schema],
        function_call="auto"
    )

    choice = response.choices[0]
    message = choice.message
    function_call = getattr(message, "function_call", None)

    if function_call:
        tool_output = get_current_time()
        messages.append({
            "role": "assistant",
            "content": None,
            "function_call": {
                "name": function_call.name,
                "arguments": function_call.arguments or "{}"
            }
        })
        messages.append({
            "role": "function",
            "name": "get_current_time",
            "content": tool_output
        })

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content

    return message.content


# Prediction:
# 1. Will calling run_agent("Convert 100 degrees Celsius to Fahrenheit") trigger a tool call?
#    No, because the query is a temperature conversion problem and does not require the current
#    date or time, so the get_current_time tool should not be invoked.
# 2. How many API calls will be made to answer this query?
#    One API call, because the model should respond directly without needing a second tool execution.

result = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print("\nAgent output:")
print(result)

# =====================================================================================
# Question 3:
# =====================================================================================

# Extended agent with both tools: get_current_time and celsius_to_fahrenheit


def run_agent_extended(prompt: str) -> str:
    """Run an agent that has access to both get_current_time and celsius_to_fahrenheit."""
    tool_schemas = [
        {
            "name": "get_current_time",
            "description": "Get the current date and time when asked.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        celsius_to_fahrenheit_schema
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to tools. "
                "Use get_current_time when the user asks for the current date or time. "
                "Use celsius_to_fahrenheit when the user asks to convert a Celsius temperature to Fahrenheit. "
                "If no tool is needed, answer directly."
            )
        },
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        functions=tool_schemas,
        function_call="auto"
    )

    choice = response.choices[0]
    message = choice.message
    function_call = getattr(message, "function_call", None)

    if function_call:
        func_name = function_call.name
        arguments = json.loads(function_call.arguments or "{}")

        if func_name == "get_current_time":
            tool_output = get_current_time()
        elif func_name == "celsius_to_fahrenheit":
            tool_output = celsius_to_fahrenheit(arguments["celsius"])
        else:
            tool_output = f"Error: unknown tool {func_name}"

        print(f"  Tool called: {func_name}")
        print(f"  Tool arguments: {arguments}")
        print(f"  Tool result: {tool_output}")

        messages.append({
            "role": "assistant",
            "content": None,
            "function_call": {
                "name": func_name,
                "arguments": function_call.arguments or "{}"
            }
        })

        messages.append({
            "role": "function",
            "name": func_name,
            "content": tool_output
        })

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content
    print("  No tools needed.")
    return message.content


response_a = run_agent_extended("What is 37 degrees Celsius in Fahrenheit?")
print("\nResponse A:", response_a)
# Tool WAS called: celsius_to_fahrenheit with celsius=37.
# The model recognized this as a temperature conversion request and used the tool.

response_b = run_agent_extended(
    "What is the boiling point of water in plain English?")
print("\nResponse B:", response_b)
# Tool was NOT called. The model already knows from training that water boils at
# 100°C / 212°F. This is general knowledge, so it answered directly without a tool.

# =====================================================================================
# Lesson 03: Multi-Tool Agent
# =====================================================================================

print('='*100)
print("---------------------- Lesson 03: Multi-Tool Agent ----------------------")
print('='*100)

# =====================================================================================
# Question 4
# =====================================================================================

print('='*100)
print(" Question 4")
print('='*100)

RESOURCES_DIR = Path("../../python-200/lessons/07_AI_agents/resources")


class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # --- Tools (public methods) --------------------------------------

    def list_csv_files(self):
        """
        List available CSV files in resources/.
        """
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.

        filename can be "bike_commute" or "bike_commute.csv".
        """
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """
        Return column names for the currently loaded CSV.
        """
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """
        Simple summary for a single column using pandas.describe().
        """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """
        Plot from the active CSV.

        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."

        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"

        # If someone accidentally passes x == y, treat it like "plot y"
        if x == y:
            x = None

        # Scatter needs x
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."

        title_csv = self.csv_name or "current CSV"

        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."

        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"

        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()

        return f"Plotted {y} vs {x} as a {plot_type}."

    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        # your code here
        error = self._ensure_loaded()
        if error:
            return error
        pearson_r, p_value = stats.pearsonr(self.df[col1], self.df[col2])

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(pearson_r, 4),
            "p_value": round(p_value, 4),
        }


print("Class defined")

csv_backend = CsvManager(RESOURCES_DIR)

node_tools = {
    "list_csv_files": csv_backend.list_csv_files,
    "load_csv": csv_backend.load_csv,
    "get_columns": csv_backend.get_columns,
    "summarize_columns": csv_backend.summarize_columns,
    "describe_column": csv_backend.describe_column,
    "plot_data": csv_backend.plot_data,
    "compute_correlation": csv_backend.compute_correlation,
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources/ folder.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources/ folder and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "CSV filename in resources/, e.g. 'bike_commute.csv'.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the column names of the currently loaded CSV.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Show basic summary statistics for columns (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names. If omitted, summarize all columns.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Show basic summary statistics for a single column (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to describe.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data from the active CSV. If only y is provided, plot y vs row index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {"type": "string", "description": "Column name for y-axis."},
                    "x": {"type": "string", "description": "Optional column name for x-axis."},
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                        "description": "Type of plot to create.",
                    },
                },
                "required": ["y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute the Pearson correlation and p_value between two numeric columns in the loaded CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string",
                        "description": "First Column name",
                    },
                    "col2": {
                        "type": "string",
                        "description": "First Column name",
                    }
                },
                "required": ["col1", "col2"]
            }
        }
    }
]


def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run through one react-agent loop using a simple tool-using agent.
    `messages` parameter will usually just contain a system prompt, 
    and then user text will be appended.  

    The loop has three main steps:

    REASON:
      - Call the model with the conversation so far.
      - The model either replies normally, or asks to call a tool from tool set.

    ACT:
      - If tools are requested, run the Python functions

    OBSERVE:
      - Append each requested tool result back into the LLMs conversation history.
      - On the next iteration, the model reads those tool call results and determines
        whether it has reached the goal.

    Stop condition:
      - If the model returns an assistant message with no tool calls, this is the 
        final answer for this react cycle, this implies that reasoning alone without 
        tool calls was enough.  
      - max_tool_rounds is a safety cap to prevent infinite loops.
    """
    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        """
        Return a tool's return value as a message that can be appended to the
        LLMs conversation history. The model will read this tool output on the next
        REASON step.
        """
        content = json.dumps(result, default=str) if not isinstance(
            result, str) else result
        tool_message = {"role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content, }
        return tool_message

    for loop_idx in range(max_tool_rounds):
        # REASON: call the model
        # Here it will make use of any previous tool outputs it appended ("observed")
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        # Append the assistant message to the conversation history.
        # Use a plain dict so `messages` stays simple and inspectable.
        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump()
                                             for tc in msg.tool_calls]
        messages.append(assistant_entry)

        # No tool calls means the model is answering directly.
        if not msg.tool_calls:
            return msg.content

        # ACT + OBSERVE: run each tool call, then append its result.
        # Note there may be multiple tool calls
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {
                        "error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}

            # OBSERVE: append the tool result back into the conversation history.
            messages.append(observe_tool_result(tool_call.id, result))

            # After we appending information about all tool outputs, we loop back and REASON again.

    return "I hit the tool-round limit. Try a simpler request."


print("run_agent_cyycle defined.")
print("compute_correlation tool added to tools_schema and node_tools")

# =====================================================================================
# Question 5
# =====================================================================================

print('='*100)
print(" Question 5")
print('='*100)

SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
)

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(
    messages, "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.")
print(result)

# =====================================================================================
# Question 6
# =====================================================================================

print('='*100)
print(" Question 6")
print('='*100)

print(json.dumps(messages, indent=2, default=str))

# Each role in the messages list maps to a part of the react loop.
# "system": the initial instructions/persona given to the model before the conversation starts.
# "user":   the human's query that kicks off a react cycle
# "assistant": the model's response: either a final answer or a request to call tool's
# "tool": the result returned by an external tool after execution

# =====================================================================================
# Lesson 04: smolagents
# =====================================================================================

print('='*100)
print(" Lesson 04: smolagents")
print('='*100)

# =====================================================================================
# Question 7
# =====================================================================================

print('='*100)
print(" Question 7")
print('='*100)


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation between two columns in the loaded CSV.

    Args:
    col1: The name of the first column.
    col2: The name of the second column.

    Returns:
    A dict with col1, col2, pearson_r, and p_value, or an error dict.
    """
    return csv_backend.compute_correlation(col1, col2)


print(compute_correlation.description)

# Comparison:
# smolagents automatically generates the tool description from the function name,
# type hints, and the Google-style docstring. The manually written JSON schema in Q4
# required us to explicitly specify "name", "description", and "parameters" (including
# each parameter's type and description) as a nested dictionary.
#
# What smolagents needs from the developer to produce a good description:
#    1. A descriptive function name (used as the tool name)
#    2. Type hints on all parameters and the return type
#    3. A clear Google-style docstring with:
#       - A summary line (becomes the tool description)
#       - An "Args:" section (becomes parameter descriptions)
#       - A "Returns:" section (documents the output)
# If any of these are missing or vague, the auto-generated description will be poor
# and the agent may not know when or how to use the tool.


# =====================================================================================
# Question 8
# =====================================================================================

print('='*100)
print(" Question 8")
print('='*100)


@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.

    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return csv_backend.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.

    Args:
        filename: CSV filename in resources/. You can pass "bike_commute" or "bike_commute.csv".

    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return csv_backend.load_csv(filename)


@tool
def get_columns() -> list[str] | dict:
    """Return column names for the currently loaded CSV.

    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return csv_backend.get_columns()


@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Return summary stats for selected columns (or all columns). 
    This includes count, mean, std, min, max, and percentiles for numeric columns,
    or count, unique, top, freq for categorical columns.

    Args:
        columns: Column names to summarize. If None, summarizes all columns.

    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """
    return csv_backend.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Describe a single column (basic stats) for the requested column.
    This includes count, mean, std, min, max, and percentiles for numeric column,
    or count, unique, top, freq for categorical column.

    Args:
        column: The name of the column to describe.

    Returns:
        A dict of basic stats for the column, or an error dict.
    """
    return csv_backend.describe_column(column)


@tool
def plot_data(y: str, x: str | None = None, plot_type: str = "line") -> str | dict:
    """Plot from the active CSV.

    Args:
        y: Column name to plot on the y-axis. 
        x: Column name to plot on the x-axis. If None, use row index.
        plot_type: "line" or "scatter". Scatter requires x and y.

    Returns:
        Generates and shows the plot. 
        Retirms a short success message string, or an error dict/string.
    """
    return csv_backend.plot_data(y=y, x=x, plot_type=plot_type)


TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
]

api_key = os.getenv("OPENAI_API_KEY")
model_to_use = "gpt-4o-mini"  # default model ID
model = OpenAIServerModel(
    api_key=api_key,
    model_id=model_to_use,
)

TOOL_AGENT_PROMT = (
    "You are a small data assistant to help analyze files stored in resources/. "
    "Use the available tools to do any work requested (do not guess). "
    "Keep answers short and student-friendly."
)

CODE_INSTRUCTIONS = """
You are a helpful CSV analysis assistant.

You can do two kinds of actions:
1) Call the provided tools.
2) Write and execute Python code when tools are not enough.

Rules:
- Prefer tools for simple tasks.
- IMPORTANT: If the user requests plot styling (color, marker, title text, labels, grid, etc.)
  that the plot_data tool cannot control, DO NOT call plot_data.
  Instead, write matplotlib code directly so the plot matches the request.
  If code execution fails, do not fall back to plot_data when the user requested styling (like color).
  Explain what failed and what you would need to proceed.
- Be honest: only claim you did something if the code or tool actually did it.
- Assume the active dataset lives in csv_manager.df after a CSV is loaded.
"""

tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=model,
    instructions=TOOL_AGENT_PROMT,
)

code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=CODE_INSTRUCTIONS,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    max_steps=8,
)

prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

print("\n--- ToolCallingAgent ---")
response_tool = tool_agent.run(prompt)
print("ToolCallingAgent response:", response_tool)

print("\n--- CodeAgent ---")
response_code = code_agent.run(prompt, additional_args={
                               "csv_manager": csv_backend})
print("CodeAgent response:", response_code)

# 1. What did each agent actually produce?
#     - ToolCallingAgent: It loaded the CSV and called plot_data to create a scatter plot,
#       but it could NOT change the dot color to green because the plot_data tool has no
#       color parameter. It either ignored the color request or falsely claimed it was done.
#     - CodeAgent: It loaded the CSV (via the tool), then wrote custom matplotlib code to
#       create a scatter plot with green dots (color='green'). It successfully fulfilled
#       the full request including the styling.
#
# 2. What does this reveal about when each type of agent is more useful?
#     - ToolCallingAgent is best for straightforward tasks that fall entirely within the
#       capabilities of the provided tools. It's simpler, faster, and less risky.
#     - CodeAgent is better when the task requires flexibility beyond what the tools offer
#       (like custom styling, new computations, or combining multiple steps in novel ways).
#       It can write its own code to fill gaps, but carries more risk of errors or hallucination.

# =====================================================================================
# Question 9
# =====================================================================================

print('='*100)
print(" Question 9")
print('='*100)

# 1. Describe a task where a ToolCallingAgent would be a better choice than a CodeAgent.
#
#     Loading a CSV file and retrieving summary statistics for its columns. This task is
#     entirely within the capabilities of the existing tools (load_csv + summarize_columns),
#     so there's no need for the agent to generate any custom code. The property that makes
#     it a good fit for a tool-based approach is that the task is well-defined, predictable,
#     and fully covered by pre-built tools — no creative or flexible logic is required.
#
# 2. What is one meaningful risk of using a CodeAgent that does not apply to a ToolCallingAgent?
#
#     A CodeAgent can generate and execute arbitrary Python code, which means it could
#     potentially run harmful operations — deleting files, making network requests, entering
#     infinite loops, or accessing sensitive data — even if unintentionally. A ToolCallingAgent
#     can only invoke the specific, pre-defined functions you provide, so its actions are
#     bounded and predictable. The CodeAgent trades safety for flexibility.

print("Comment Added")
