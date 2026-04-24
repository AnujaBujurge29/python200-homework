import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split


import os
os.system('cls')
# ------------------------------------------------------------------------------------------------------
#                                                Week 2
#                                           Part 1: Warmup Exercises
# ------------------------------------------------------------------------------------------------------

print('-'*100)
print("--------------- Part 1: Warmup Exercises ---------------")
print('-'*100)
# ------------------------------------------------------------------------------------------------------
#                                           scikit-learn
# ------------------------------------------------------------------------------------------------------

print('\n')
print('-'*100)
print("--------------- scikit-learn ---------------")
print('-'*100)
# -------------------------------------------------------------------------------------------------------
# scikit-learn Question 1
# -------------------------------------------------------------------------------------------------------

print("\n-------------------------------Question 1: scikit-learn --> create → fit → predict------------------")

years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

# Create the model
model = LinearRegression()

# Fit the model to the data
model.fit(years, salary)

# Predict salaries for 4 and 8 years of experince
years_to_predict = np.array([[4], [8]])
predictions = model.predict(years_to_predict)

print(f"Slope - Coefficient: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Predicted salary for 4 years of experince: {predictions[0]:.2f}")
print(f"Predicted salary for 8 years of experince: {predictions[1]:.2f}")

# -------------------------------------------------------------------------------------------------------
# scikit-learn Question 2:
# -------------------------------------------------------------------------------------------------------

print("\n-------------------------------Question 2: scikit-learn --> Reshaping arrays------------------")
x = np.array([10, 20, 30, 40, 50])
print(f"Original shape of array: {x.shape}")

x_2d = x.reshape(-1, 1)
print(f"Reshaped array: {x_2d.shape}")

# Explaination:
# scikit-learn requires X to be 2D because the API is designed to handle multiple
# features (columns) for each sample (row). Even with a single feature, we need a
# 2D array of shape (n_samples, n_features) where n_features=1. This consistent
# interface allows the same model code to work whether you have 1 feature or 100.
# The reshape(-1, 1) converts our 1D array of 5 values into a 2D array with 5 rows
# and 1 column, where each row represents one sample.


# -------------------------------------------------------------------------------------------------------
# scikit-learn Question 3:
# -------------------------------------------------------------------------------------------------------

print("\n-------------------------------Question 3: scikit-learn --> K-Means Clustering------------------")
X_clusters, _ = make_blobs(n_samples=120, centers=3,
                           cluster_std=0.8, random_state=7)

# Create K-Means Model
k_means = KMeans(n_clusters=3, random_state=42)

# Fir K-Menas Model
# k_means.fit(X_clusters)

# Predict Cluster Label
labels = k_means.fit_predict(X_clusters)
# labels = k_means.predict(X_clusters)

print(f"Cluster Centers: {k_means.cluster_centers_}")
print(f"Points per Cluster: {np.bincount(labels)}")

# create a scatter plot

plt.figure(figsize=(8, 6))
plt.scatter(X_clusters[:, 0], X_clusters[:, 1],
            c=labels, cmap='viridis', alpha=0.7)

plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1],
            c='black', marker='X', s=200, label='cluster center')
plt.title("K-Mean Clustering - 3 clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.savefig('outputs/kmeans_clusters.png')
plt.close()
print("Saved: kmeans_clusters.png'")


# ------------------------------------------------------------------------------------------------------
#                                           Linear Regression
# ------------------------------------------------------------------------------------------------------
print('-'*100)
print("--------------- Linear Regression ---------------")
print('-'*100)

# Generate synthetic medical costs dataset
np.random.seed(42)
num_patients = 100
age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# -------------------------------------------------------------------------------------------------------
# Linear Regression Question 1
# -------------------------------------------------------------------------------------------------------

print("\n-------------------------------Question 1: Linear Regression --> scatter plot of cost vs age------------------")
plt.figure(figsize=(10, 6))
scatter = plt.scatter(age, cost, c=smoker, cmap='coolwarm', alpha=0.7)
plt.colorbar(scatter, label='Smoker (0=No, 1=Yes)')
plt.title('Medical Cost vs Age')
plt.xlabel('Age in years')
plt.ylabel('Annual Medical Cost ($)')
plt.savefig('outputs/cost_vs_age.png')
plt.close()
print("Saved: cost_vs_age.png")

# --------------------------------------------------------------------------------------------
# Qu: Are there two distinct groups visible? What does that suggest about the smoker variable?
# Ans: Looking at the scatter plot, there are clearly two distinct groups visible.
# The non-smokers (blue) cluster in the lower range of costs, while smokers (red)
# form a separate, higher cluster. Both groups show a positive relationship between
# age and cost, but the smoker group is shifted upward by a significant amount.
# This suggests the smoker variable has a strong effect on medical costs - it adds
# a large "offset" to the cost independent of age.
# --------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------------
# Linear Regression Question 2
# -------------------------------------------------------------------------------------------------------

print("\n-------------------------------Question 2: Linear Regression --> Train and test - splits------------------")

# Reshape age to 2D
X = age.reshape(-1, 1)
Y = cost

# Splits - 80% train and 20% test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

print(f"X_train Shape: {X_train.shape}")
print(f"X_test Shape: {X_test.shape}")
print(f"Y_train Shape: {Y_train.shape}")
print(f"Y_test Shape: {Y_test.shape}")

# -------------------------------------------------------------------------------------------------------
# Linear Regression Question 3
# -------------------------------------------------------------------------------------------------------

print("\n-------------------------------Question 3: Linear Regression --> Fit Model with Age------------------")
model_age = LinearRegression()
model_age.fit(X_train, Y_train)

# Make Predictions
Y_pred = model_age.predict(X_test)

# Calulate matrix
rmse = np.sqrt(np.mean((Y_pred - Y_test) ** 2))
r2 = model_age.score(X_test, Y_test)

print(f"Slope: {model_age.coef_[0]:.2f}")
print(f"Intercept: {model_age.intercept_:.2f}")
print(f"Test RMSE: ${rmse:,.2f}")
print(f"Test R(square): {r2:.4f}")

# --------------------------------------------------------------------------------------------
# Qu: what does it mean for medical costs?
# Ans: The slope of ~200 means that for each additional year of age, medical costs
# increase by approximately $200 on average. However, the low R² (around 0.18)
# indicates that age alone explains only about 18% of the variation in medical
# costs - there's a lot of unexplained variance, which makes sense because we
# know smoker status has a huge impact that we haven't included yet.
# --------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------
# Linear Regression Question 4
# -------------------------------------------------------------------------------------------------------

print("\n-------------------------------Question 4: Linear Regression --> Two feature model (age + smoker------------------")

# Create feature matrix with age and smoker
X_full = np.column_stack([age, smoker])

# Split the data
X_train_full, X_test_full, Y_train_full, Y_test_full = train_test_split(
    X_full, Y, test_size=0.2, random_state=42)

# Fit the model
model_full = LinearRegression()
model_full.fit(X_train_full, Y_train_full)

# Calculate R²
r2_full = model_full.score(X_test_full, Y_test_full)

print(f"Test R(square) (age only): {r2:.4f}")
print(f"Test R(square) (age + smoker): {r2_full:.4f}")
print(f"Age coefficient: {model_full.coef_[0]:.2f}")
print(f"Smoker coefficient: {model_full.coef_[1]:.2f}")


# --------------------------------------------------------------------------------------------
# Qu: what does it represent in practical terms?
# Ans:
# The smoker coefficient of approximately $15,000 represents the additional
# annual medical cost associated with being a smoker, holding age constant.
# In practical terms, a smoker pays about $15,000 more per year in medical
# costs than a non-smoker of the same age. Adding the smoker flag dramatically
# improved R² (from ~0.18 to ~0.82), confirming that smoking status is a
# powerful predictor of medical costs. This makes sense given how the data
# was generated: cost = 200*age + 15000*smoker + noise.
# --------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------
# Linear Regression Question 5
# -------------------------------------------------------------------------------------------------------

print("\n-------------------------------Question 5: Linear Regression --> Predicted vs Actual Plot------------------")
# Get predictions from the two-feature model
Y_pred_full = model_full.predict(X_test_full)

plt.figure(figsize=(8, 8))
plt.scatter(Y_pred_full, Y_test_full, alpha=0.7, c='steelblue')

# Add diagonal reference line
min_val = min(Y_pred_full.min(), Y_test_full.min())
max_val = max(Y_pred_full.max(), Y_test_full.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--',
         linewidth=2, label='Perfect Prediction')

plt.title('Predicted vs Actual')
plt.xlabel('Predicted Cost ($)')
plt.ylabel('Actual Cost ($)')
plt.legend()
plt.savefig('outputs/predicted_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved predicted vs actual plot to outputs/predicted_vs_actual.png")

# --------------------------------------------------------------------------------------------
# Qu: wwhat does it mean when a point falls above the diagonal? What about below?
# Ans:
# When a point falls ABOVE the diagonal, it means the actual value is higher than
# the predicted value - the model UNDERESTIMATED the cost for that patient.
# When a point falls BELOW the diagonal, it means the actual value is lower than
# the predicted value - the model OVERESTIMATED the cost for that patient.
# Points on the diagonal represent perfect predictions where predicted = actual.
# --------------------------------------------------------------------------------------------
