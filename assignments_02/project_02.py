import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import seaborn as sns
from pathlib import Path
from prefect import task, get_run_logger, flow
from scipy import stats


import os
os.system('cls')

os.makedirs("outputs", exist_ok=True)
# --------------------------------------------------------------------------------
#                    Task 1:  Load and Explore
# --------------------------------------------------------------------------------
print('-' * 100)
print("Task 1: Load and Explore")
print('-' * 100)


df = pd.read_csv('student_performance_math.csv',
                 sep=";", encoding="utf-8", decimal=",")

print(f"Dataset Shape: {df.shape}")
# print(f"First 5 rows: {df.head()}")
# print(f"Data Type:\n{df.dtypes}")

# Plot histogram of G3 (final grades)
plt.figure(figsize=(10, 6))
plt.hist(df['G3'], bins=21, range=(-0.5, 20.5), edgecolor='black', alpha=0.7)
plt.title('Distribution of Final Math Grades')
plt.xlabel('Final Grade (G3)')
plt.ylabel('Frequency')
plt.xticks(range(0, 21))
plt.savefig('outputs/g3_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved histogram to outputs/g3_distribution.png")


# --------------------------------------------------------------------------------
#                    Task 2:  Preprocess the Data
# --------------------------------------------------------------------------------
print('-' * 100)
print("Task 2: Preprocess the Data")
print('-' * 100)

print(f"Shape before filtering G3=0: {df.shape}")

# Filter out rows where G3=0 (students who didn't take the exam)
df_clean = df[df['G3'] != 0].copy()

print(f"Shape after filtering G3=0: {df_clean.shape}")
print(f"Rows removed: {len(df) - len(df_clean)}")

# ---------------------------------------------------------------------------------
# Qu: why would keeping these rows distort the model?
# Ans:
# Keeping G3=0 rows would distort the model because these aren't actual grades -
# they represent absence from the final exam. Treating "didn't show up" as the
# same as "scored zero" would conflate two very different things. A student who
# scored 0 might have struggled academically, while a student who was absent
# might have dropped out, moved away, or had other non-academic reasons. Their
# feature patterns (absences, studytime, etc.) may be completely different from
# students who actually took and failed the exam.
# ---------------------------------------------------------------------------------

# Convert yes/no columns to 1/0
binary_cols = ['schoolsup', 'internet', 'higher', 'activities']
for col in binary_cols:
    df_clean[col] = df_clean[col].map({'yes': 1, 'no': 0})
    # Also convert in original for correlation comparison
    df[col] = df[col].map({'yes': 1, 'no': 0})

# Convert sex column: F=0, M=1
df_clean['sex'] = df_clean['sex'].map({'F': 0, 'M': 1})
df['sex'] = df['sex'].map({'F': 0, 'M': 1})

print("\nBinary columns converted (yes/no → 1/0, F/M → 0/1)")

# Compare Pearson correlation between absences and G3 on both datasets
corr_original = df['absences'].corr(df['G3'])
corr_filtered = df_clean['absences'].corr(df_clean['G3'])

print(f"\nPearson correlation (absences vs G3):")
print(f"Original dataset (with G3=0): {corr_original:.4f}")
print(f"Filtered dataset (no G3=0): {corr_filtered:.4f}")

# ---------------------------------------------------------------------------------
# Qu:
# why filtering changes the result: what were students with G3=0 doing in the original data that made absences look like a weak predictor?
# Ans:
# In the original dataset, students with G3=0 (who missed the final exam) often
# have very high absences throughout the year - they were chronically absent.
# But G3=0 isn't a "grade" - it's a non-event. When we include these students,
# we get a cluster of (high absences, G3=0) points that pull the correlation
# in a strange direction. The filtered correlation is more meaningful because
# it captures the relationship between absences and actual academic performance.
# Students who show up but miss class occasionally may still learn; students
# who stop showing up entirely are a different population entirely.
# ---------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
#                    Task 3: Exploratory Data Analysis
# --------------------------------------------------------------------------------
print('-' * 100)
print("Task 3: Exploratory Data Analysis")
print('-' * 100)

# Numeric features to analyze
numeric_features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'absences',
                    'freetime', 'goout', 'Walc', 'schoolsup', 'internet', 'higher', 'activities', 'sex']

# Compute Pearson correlation
correlation = {}
for feature in numeric_features:
    if feature in df_clean.columns:
        correlation[feature] = df_clean[feature].corr(df_clean['G3'])

# Sort
sorted_corr = sorted(correlation.items(), key=lambda x: x[1])

print(f"Sorted Pearson correlations with G3:\n")
for feature, corr in sorted_corr:
    print(f"  {feature:12s}: {corr:.4f}")

# Visualization 1: Failures vs G3 scatter plot
plt.figure(figsize=(8, 6))
# Add jitter so points don't overlap completely
jittered_failures = df_clean['failures'] + \
    np.random.normal(0, 0.1, len(df_clean))
plt.scatter(jittered_failures, df_clean['G3'], alpha=0.5)
plt.title('Failures vs Final Grade (G3)')
plt.xlabel('Number of Past Failures')
plt.ylabel('Final Grade (G3)')
plt.xticks([0, 1, 2, 3])
plt.savefig('outputs/failures_vs_g3.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved visualization: outputs/failures_vs_g3.png")

# This plot clearly shows that students with 0 failures have a wide range
# of grades (clustering higher), while students with 1-3 failures tend to cluster
# at lower grades. The negative correlation is visually evident.

# Visualization 2: Study time and higher education aspiration effects
plt.figure(figsize=(10, 6))
colors = df_clean['higher'].map({1: 'green', 0: 'red'})
plt.scatter(df_clean['studytime'], df_clean['G3'], c=colors, alpha=0.5)
plt.title('Study Time vs Final Grade (colored by Higher Ed. Aspirations)')
plt.xlabel('Weekly Study Time (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)')
plt.ylabel('Final Grade (G3)')
plt.scatter([], [], c='green', label='Wants Higher Ed. (1)')
plt.scatter([], [], c='red', label='No Higher Ed. Goals (0)')
plt.legend()
plt.savefig('outputs/studytime_higher_vs_g3.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved visualization: outputs/studytime_higher_vs_g3.png")

# This plot shows that students who want to pursue higher education (green)
# tend to cluster at higher grades across all study time levels. The combination of
# motivation (higher) and effort (studytime) seems to matter for academic success.


# --------------------------------------------------------------------------------
#                    Task 4: Baseline Model
# --------------------------------------------------------------------------------
print('-' * 100)
print("Task 4: Baseline Model")
print('-' * 100)

# Use only failures as the feature
X_baseline = df_clean[['failures']].values
Y = df_clean['G3'].values

# Split the data
X_train_b, X_test_b, Y_train_b, Y_test_b = train_test_split(
    X_baseline, Y, test_size=0.2, random_state=42)

# Fit the model
model_baseline = LinearRegression()
model_baseline.fit(X_train_b, Y_train_b)

# Predictions and metrics
Y_pred_b = model_baseline.predict(X_test_b)
rmse_baseline = np.sqrt(np.mean((Y_pred_b - Y_test_b) ** 2))
r2_baseline = model_baseline.score(X_test_b, Y_test_b)

print(f"Slope (failures coefficient): {model_baseline.coef_[0]:.3f}")
print(f"Intercept: {model_baseline.intercept_:.3f}")
print(f"Test RMSE: {rmse_baseline:.3f}")
print(f"Test R(square): {r2_baseline:.4f}")

# what do the slopes and RMSE tell you:
#
# On a 0-20 scale, the slope of about -1.9 means each additional past failure is
# associated with roughly 2 points lower on the final grade. The RMSE of ~3.1 means
# our typical prediction error is about 3 points on a 20-point scale - equivalent
# to about 15% of the grade range.
# The R² of ~0.15 is somewhat low but not unexpected from EDA - we saw failures
# has the strongest correlation, but it was still only about -0.35. A single
# variable won't explain most of the variance in something as complex as grades.

# --------------------------------------------------------------------------------
#                    Task 5: Build the Full Model
# --------------------------------------------------------------------------------
print('-' * 100)
print("Task 5: Build the Full Model")
print('-' * 100)

feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]
X = df_clean[feature_cols].values
Y = df_clean["G3"].values

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

# Fit the model
model_full = LinearRegression()
model_full.fit(X_train, Y_train)

# Metrics
train_r2 = model_full.score(X_train, Y_train)
test_r2 = model_full.score(X_test, Y_test)
Y_pred = model_full.predict(X_test)
rmse_full = np.sqrt(np.mean((Y_pred - Y_test) ** 2))

print(f"Train R²: {train_r2:.4f}")
print(f"Test R²:  {test_r2:.4f}")
print(f"Test RMSE: {rmse_full:.3f}")
print(f"\nComparison to baseline:")
print(f"Baseline Test R²: {r2_baseline:.4f}")
print(f"Full Model Test R²: {test_r2:.4f}")
print(f"Improvement: {test_r2 - r2_baseline:.4f}")

print("\nFeature Coefficients:")
print("-" * 30)
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:12s}: {coef:.3f}")


#   surprising results:
# - The sex coefficient (positive for male=1) is interesting. In this Portuguese
#    dataset from 2005, boys appear to have a slight math advantage. As the
#    assignment notes, research shows this varies by country and correlates with
#    gender equality measures, suggesting social factors rather than inherent ability.

# - schoolsup (extra school support) might have a surprising sign - if positive,
#    it could mean struggling students who got help improved; if negative, it
#    might just be because struggling students are the ones who need support.

# - activities being negative or near-zero is somewhat surprising - one might
#    expect extracurriculars to indicate a well-rounded student.

# train vs test R²:
# If the gap between train R² and test R² is small, the model generalizes well.
# If train R² is much higher than test R², we might be overfitting - the model
# learned patterns specific to the training data that don't hold in new data.

# feature selection for production:
# For production, I would keep: failures (strongest predictor), higher (strong
# motivation signal), Medu (parent education matters), studytime
# I might drop: activities, traveltime (weak effect), sex (ethical concerns
# about using sex as a predictor even if statistically significant)
# The choice depends on both predictive power AND whether it's appropriate/ethical
# to use certain features in a real-world educational setting.

# --------------------------------------------------------------------------------
#                    Task 6: Evaluate and Summarize
# --------------------------------------------------------------------------------
print('-' * 100)
print("Task 6: Evaluate and Summarize")
print('-' * 100)

# predicted vs actaul plot
plt.figure(figsize=(8, 8))
plt.scatter(Y_pred, Y_test, alpha=0.6, c='steelblue')

# Add a diagonal reference line
min_val = min(Y_pred.min(), Y_test.min())
max_val = max(Y_pred.max(), Y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

plt.title('Predicted vs Actual (Full Model)')
plt.xlabel('Predicted Grade')
plt.ylabel('Actual Grade')
plt.legend()
plt.savefig('outputs/predicted_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved plot to outputs/predicted_vs_actual.png")

# does the model seem to struggle more at the high end, the low end, or 
# is error roughly uniform across grade levels? 
# What does a value above or below the diagonal mean?

# Points above the diagonal: actual > predicted → model UNDERESTIMATED
# Points below the diagonal: actual < predicted → model OVERESTIMATED
# Looking at the distribution, error seems relatively uniform across grade levels,
# though there may be more scatter at the extremes. The model tends to predict
# values in a narrower range than the actual grades (regression to the mean).

# SUMMARY:
# From the predicted vs actual plot:
# The model tends to struggle more at the high and low ends,
# where predictions are less accurate compared to middle grades.
# Most points are scattered around the diagonal, so error is somewhat uniform,
# but there is slightly more spread at extreme values.

# Points above the diagonal mean the actual grade is higher than predicted
# (the model underestimated the student's performance).
# Points below the diagonal mean the predicted grade is higher than actual
# (the model overestimated the student's performance).


# --------------------------------------------------------------------------------
#                    Neglected Feature: The Power of G1
# --------------------------------------------------------------------------------
print('-' * 100)
print("Neglected Feature: The Power of G1")
print('-' * 100)

# Add G1 to the feature set
feature_cols_g1 = feature_cols + ["G1"]
X_g1 = df_clean[feature_cols_g1].values

# Split
X_train_g1, X_test_g1, Y_train_g1, Y_test_g1 = train_test_split(X_g1, Y, test_size=0.2, random_state=42)

# Fit
model_g1 = LinearRegression()
model_g1.fit(X_train_g1, Y_train_g1)

# Evaluate
test_r2_g1 = model_g1.score(X_test_g1, Y_test_g1)
Y_pred_g1 = model_g1.predict(X_test_g1)
rmse_g1 = np.sqrt(np.mean((Y_pred_g1 - Y_test_g1) ** 2))

print(f"Test R² without G1:  {test_r2:.4f}")
print(f"Test R² with G1:     {test_r2_g1:.4f}")
print(f"Improvement:         {test_r2_g1 - test_r2:.4f}")
print(f"Test RMSE with G1:   {rmse_g1:.3f}")

# does a high R² here mean G1 is causing G3? 
# Correlation does not mean one thing causes the other.
# G1 and G3 are both just measuring how good a student is at math.
# That’s why they are related.
# Both are influenced by things like study habits, past learning, and home environment.
# So G1 does not cause G3 — they both come from the same underlying factors.


# Is this a useful model for identifying students who might struggle? 
# This model is only useful after we have G1 scores.
# But once we already have G1, we can directly see which students are struggling.
# So the model doesn’t really add new information — it just repeats what G1 already shows.


# What might educators need to do if they wanted to intervene early, before G1 is even available?
# If we want to help students before G1 is available, we cannot use G1.
# That’s why we removed it earlier.
# Instead, we use information we already have at the start, like past failures,
# parents' education, and signs of motivation.
# This model is less accurate, but it is more useful because it helps us act early.
# Teachers can use it to identify students who may need extra help from the beginning,
# even before any test scores are available.
