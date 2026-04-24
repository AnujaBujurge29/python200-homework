import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import os
os.system('cls')

# =====================================================================================
#                    Assignment 1 - Part 1: Warm up Exercise
# =====================================================================================

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}

# =====================================================================================
#                                     Panda Review
# =====================================================================================

# Pandas Question 1:
print("\n----------------------Pandas Question 1--------------------------------------")
df = pd.DataFrame(data)
print(f"First 3 rows:\n{df.head(3)}")
print("--------------------------------------------------")
print(f"Shape:\n{df.shape}")
print("--------------------------------------------------")
print(f"Data Types:\n{df.dtypes}")

# Pandas Question 2:
print("\n----------------------Pandas Question 2--------------------------------------")
filter_data = df[(df["passed"]) & (df["grade"] > 80)]
print(f"Students who passed and have a grade above 80:\n{filter_data}")

# Pandas Question 3:
print("\n----------------------Pandas Question 3--------------------------------------")
df["grade_curved"] = df["grade"] + 5
print(f"New data:\n{df}")

# Pandas Question 4:
print("\n----------------------Pandas Question 4--------------------------------------")
df["name_upper"] = df["name"].str.upper()
print(f"New data:\n{df}")

# Pandas Question 5:
print("\n----------------------Pandas Question 5--------------------------------------")
mean_by_city = df.groupby("city")["grade"].mean()
print(f"Mean grade for each city:\n{mean_by_city}")

# Pandas Question 6:
print("\n----------------------Pandas Question 6--------------------------------------")
df["city"] = df["city"].replace("Austin", "Houston")
print(df[["name", "city"]])

# Pandas Question 7:
print("\n----------------------Pandas Question 7--------------------------------------")
sorted_df = df.sort_values(by="grade", ascending=False)
print(f"Sorted DF:\n{sorted_df.head()}")
print(f"\nSorted DF first 3 rows:\n{sorted_df.head(3)}")

# =====================================================================================
#                               NumPy Review
# =====================================================================================

# NumPy Question 1:
print("\n----------------------NumPy Question 1--------------------------------------")
arr_1D = np.array([10, 20, 30, 40, 50])
print(f"Shape of 1D array: {arr_1D.shape}")
print(f"Data type of 1D array: {arr_1D.dtype}")
print(f"Dimension of array: {arr_1D.ndim}")

# NumPy Question 2:
print("\n----------------------NumPy Question 2--------------------------------------")
arr_2D = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print(f"Shape of array: {arr_2D.shape}")
print(f"Data type of 1D array: {arr_2D.size}")

# NumPy Question 3:
print("\n----------------------NumPy Question 3--------------------------------------")
print(f"Slice of 2D array:\n{arr_2D[:2, :2]}")

# NumPy Question 4:
print("\n----------------------NumPy Question 4--------------------------------------")
zeros = np.zeros((3, 4))
ones = np.ones((2, 5))
print(f"Array of Zeros:\n{zeros}")
print(f"Array of Ones:\n{ones}")

# NumPy Question 5:
print("\n----------------------NumPy Question 5--------------------------------------")
arr_5 = np.arange(0, 50, 5)
print(f"New array: {arr_5}")
print(f"Shape: {arr_5.shape}")
print(f"Mean: {arr_5.mean()}")
print(f"Sum: {arr_5.sum()}")
print(f"Std Deviation: {arr_5.std()}")

# NumPy Question 6:
print("\n----------------------NumPy Question 6--------------------------------------")
rand_vals = np.random.normal(0, 1, 200)
print(f"Mean: {rand_vals.mean()}")
print(f"Std Deviation: {rand_vals.std()}")

# =====================================================================================
#                               Matplotlib Review
# =====================================================================================

# Matplotlib Question 1:
print("\n----------------------Matplotlib Question 1--------------------------------------")
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]
plt.figure(1)
plt.plot(x, y)
plt.title("Squares")
plt.xlabel("X")
plt.ylabel("Y")
# plt.show(block=False)

# Matplotlib Question 2:
print("\n----------------------Matplotlib Question 2--------------------------------------")
subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]
plt.figure(2)
plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("Subject")
plt.ylabel("Scores")
# plt.show(block=False)

# Matplotlib Question 3:
print("\n----------------------Matplotlib Question 3--------------------------------------")
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
plt.figure(3)
plt.scatter(x1, y1, label="Dataset 1")
plt.scatter(x2, y2, label="Dataset 2")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
# plt.show(block=False)

# Matplotlib Question 4:
print("\n----------------------Matplotlib Question 4--------------------------------------")
plt.figure(4)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(x, y)
axes[0].set_title("Squares")

axes[1].bar(subjects, scores)
axes[1].set_title("Subject Scores")

plt.tight_layout()
# plt.show(block=False)


# =====================================================================================
#                               Descriptive Statistics Review
# =====================================================================================

# Descriptive Stats Question 1:
print("\n----------------------Descriptive Stats Question 1--------------------------------------")
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
arr = np.array(data)

print(f"Mean: {arr.mean()}")
print(f"Median: {np.median(arr)}")
print(f"Variance: {arr.var()}")
print(f"Std Deviation: {arr.std()}")

# Descriptive Stats Question 2:
print("\n----------------------Descriptive Stats Question 2--------------------------------------")
scores_dist = np.random.normal(65, 10, 500)
plt.figure(5)
plt.hist(scores_dist, bins=20)
plt.title("Distribution of Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")
# plt.show(block=False)

# Descriptive Stats Question 3:
print("\n----------------------Descriptive Stats Question 3--------------------------------------")
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]
plt.figure(6)
plt.boxplot([group_a, group_b], tick_labels=["Group A", "Group B"])
plt.title("Score Comparison")
# plt.show(block=False)

# Descriptive Stats Question 4:
print("\n----------------------Descriptive Stats Question 4--------------------------------------")
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

plt.figure(7)
plt.boxplot([normal_data, skewed_data], tick_labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
# plt.show(block=False)

# Descriptive Stats Question 5:
print("\n----------------------Descriptive Stats Question 5--------------------------------------")
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

print(f"Data 1: {data1}")
print(f"Mean: {np.mean(data1)}")
print(f"Median: {np.median(data1)}")
print(f"Mode: {stats.mode(data1, keepdims=True)[0][0]}")

print(f"\nData 2: {data2}")
print(f"Mean: {np.mean(data2)}")
print(f"Median: {np.median(data2)}")
print(f"Mode: {stats.mode(data2, keepdims=True)[0][0]}")

# The mean of data2 is much higher because of the extreame data 150
# while median ignores extreme values


# =====================================================================================
#                               Hypothesis Testing Review
# =====================================================================================

# Hypothesis Question 1:
print("\n----------------------Hypothesis Question 1--------------------------------------")
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]
t_stats, p_val = stats.ttest_ind(group_a, group_b)
print(f"t-Statitics: {t_stats}")
print(f"p-Value: {p_val}")

# Hypothesis Question 2:
print("\n----------------------Hypothesis Question 2--------------------------------------")
if p_val < 0.05:
    print("The result is Statistically significant.")
else:
    print("The result is NOT Statistically significant.")

# Hypothesis Question 3:
print("\n----------------------Hypothesis Question 3--------------------------------------")
before = [60, 65, 70, 58, 62, 67, 63, 66]
after = [68, 70, 76, 65, 69, 72, 70, 71]
t_statistic, p_value = stats.ttest_rel(before, after)
print(f"t-statistic: {t_statistic}")
print(f"p-value: {p_value}")

# Hypothesis Question 4:
print("\n----------------------Hypothesis Question 4--------------------------------------")
scores = [72, 68, 75, 70, 69, 74, 71, 73]
t_statistic, p_value = stats.ttest_1samp(scores, 70)
print(f"t-statistic: {t_statistic}")
print(f"p-value: {p_value}")

# Hypothesis Question 5:
print("\n----------------------Hypothesis Question 5--------------------------------------")
t_stats, p_val = stats.ttest_ind(group_a, group_b, alternative="less")
print(f"One tailed p_value: {p_val}")


# Hypothesis Question 6:
print("\n----------------------Hypothesis Question 6--------------------------------------")
print("Group A scored significantly lower than Group B, and the"
      "difference is unlikely due to chance")

# =====================================================================================
#                               Correlation Review
# =====================================================================================

# Correlation Question 1:
print("\n----------------------Correlation  Question 1--------------------------------------")
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
corr_matrix = np.corrcoef(x, y)
print(f"Correlation Coeficient: {corr_matrix}")

# Correlation Question 2:
print("\n----------------------Correlation  Question 2--------------------------------------")
x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]
corr, p_value = stats.pearsonr(x, y)
print(f"Correlation: {corr}")
print(f"p-value: {p_value}")

# Correlation Question 3:
print("\n----------------------Correlation  Question 3--------------------------------------")
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
print(f"People correlation Matrix: {df.corr()}")

# Correlation Question 4:
print("\n----------------------Correlation  Question 4--------------------------------------")
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]
plt.figure(8)
plt.scatter(x, y)
plt.title("Negative Correlation")
plt.xlabel("X")
plt.ylabel("Y")
# plt.show(block=False)

# Correlation Question 5:
print("\n----------------------Correlation  Question 5--------------------------------------")
plt.figure(9)
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
# plt.show(block=False)
plt.show()

# =====================================================================================
#                               Pipelines
# =====================================================================================

# Pipeline Question 1:
print("\n----------------------Pipeline  Question 1--------------------------------------")
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan,
               18.0, 14.0, 16.0, 22.0, np.nan, 13.0])


def create_series(arr):
    return pd.Series(arr, name="values")


def clean_data(series):
    return series.dropna()


def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }


def data_pipeline(arr):
    series = create_series(arr)
    clean = clean_data(series)
    return summarize_data(clean)


summary = data_pipeline(arr)

for key, value in summary.items():
    print(f"{key}: {value}")

# Pipeline Question 2:
print("\n----------------------Pipeline  Question 2--------------------------------------")
