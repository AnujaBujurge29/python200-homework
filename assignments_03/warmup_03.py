import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import os
os.system('cls')

# =====================================================================================
#                    Assignment 3 - Part 1: Warm up Exercise
# =====================================================================================

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# =====================================================================================
#                                     Preprocessing
# =====================================================================================

# Preprocessing Question 1:
print("\n")
print("-" * 100)
print("Preprocessing Question 1")
print("-" * 100)

# Split - train and test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print(f"X_train Shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test Shape: {y_test.shape}")

# Preprocessing Question 2:
print("\n")
print("-" * 100)
print("Preprocessing Question 2")
print("-" * 100)

# Standard scalar
scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

print(f"Mean of each column in X_train_scaled: {X_train_scaled.mean(axis=0)}")

# We fit the scaler on X_train only to prevent data leakage.
# The test set should use transform() only so it is scaled with the same
# parameters learned from the training data.

# =====================================================================================
#                                     KNN
# =====================================================================================

# KNN Question 1:
print("\n")
print("-" * 100)
print("KNN Question 1 - unscaled")
print("-" * 100)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print("-" * 100)
print(classification_report(y_test, y_pred_knn, target_names=iris.target_names))

# KNN Question 2:
print("-" * 100)
print("KNN Question 2 - scaled")
print("-" * 100)

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn_scaled):.4f}")

# For the Iris dataset, scaling typically makes little or no diffrence because
# the four features are already on roughly similar numeric scales.
# When the features have very different magnitudes, scaling can matter a lot for knn sincce
# it relies on distnace calculations.

# KNN Question 3:
print("\n")
print("-" * 100)
print("KNN Question 3 - cross validation")
print("-" * 100)

cv_scores = cross_val_score(KNeighborsClassifier(
    n_neighbors=5), X_train, y_train, cv=5)
print(f"Fold Scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.4f}")
print(f"Std:  {cv_scores.std():.4f}")

# KNN Question 4:
print("\n")
print("-" * 100)
print("KNN Question 4 - k-values")
print("-" * 100)

k_values = [1, 3, 5, 7, 9, 11, 13, 15]
best_k, best_score = None, 0

for k in k_values:
    scores = cross_val_score(KNeighborsClassifier(
        n_neighbors=k), X_train, y_train, cv=5)
    mean_score = scores.mean()
    print(f"k={k:2d}   mean CV accuracy: {mean_score:.4f}")
    if mean_score > best_score:
        best_k, best_score = k, mean_score

print(f"Best k: {best_k}")
print(f"Mean of accuracy: {best_score:.4f}")

# I would choose the k with the hightest mean CV accuracy. If several k values tie,
# I would pick the largest k because it gives a smoother, less noisy decision boundary
# and is less likely to overfit

# =====================================================================================
#                                     Classifier Evaluation
# =====================================================================================

# Classifier Evaluation Question 1:
print("\n")
print("-" * 100)
print("Classifier Evaluation Question 1 - confusion matrix")
print("-" * 100)

con_matrix = confusion_matrix(y_test, y_pred_knn)
disp = ConfusionMatrixDisplay(
    confusion_matrix=con_matrix, display_labels=iris.target_names)
fig, ax = plt.subplots()
disp.plot(ax=ax)
fig.savefig("outputs/knn_confusion_matrix.png")
plt.close(fig)

# The model makes mistakes, it's usully mixing up versicolor and virginica.

# =====================================================================================
#                                The sklearn API: Decision Trees
# =====================================================================================

# Decision Trees Question 1:
print("\n")
print("-" * 100)
print("Decision Trees Question 1")
print("-" * 100)

dc = DecisionTreeClassifier(max_depth=3, random_state=42)
dc.fit(X_train, y_train)
y_pred_DT = dc.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_DT):.4f}")
print(classification_report(y_test, y_pred_DT, target_names=iris.target_names))

# The Decision Tree accuracy is comparable to KNN on this small dataset.
# Decision Trees split on feature thresholds (greater/less than), so the actual
# scale of the features does not matter -- scaled vs. unscaled data would
# produce the same tree structure and the same predictions.

# =====================================================================================
#                             Logistic Regression and Regularization
# =====================================================================================

# Logistic Regression Question 1:
print("-" * 100)
print("Logistic Regression Question 1")
print("-" * 100)

for C_val in [0.01, 1.0, 100]:
    lr = LogisticRegression(C=C_val, max_iter=1000, solver='lbfgs')
    # ValueError: The 'liblinear' solver does not support multiclass classification (n_classes >= 3).
    # Either use another solver or wrap the estimator in a OneVsRestClassifier to keep applying a one-versus-rest scheme.

    # So changedc'liblinear' to 'lbfgs'

    lr.fit(X_train_scaled, y_train)
    coef_sum = np.abs(lr.coef_).sum()
    print(f"C = {C_val:6.2f}  Total - COEF = {coef_sum:.4f}")

# As C increases, the total coefficient magnitude grows.  A smaller C means
# stronger regularization, which penalizes large coefficients and forces them
# toward zero.  This tells us regularization controls model complexity by
# shrinking the weights -- stronger regularization (lower C) produces simpler
# models that are less likely to overfit.

# =====================================================================================
#                                       PCA
# =====================================================================================

digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images = digits.images  # same data shaped as 8x8 images for plotting

# PCA Question 1:
print("\n")
print("-" * 100)
print("PCA: Question 1")
print("-" * 100)

print(f"X_digits shape: {X_digits.shape}")
print(f"images Shape: {images.shape}")

fig, ax = plt.subplots(1, 10, figsize=(12, 1.5))
for digit in range(10):
    idx = np.where(y_digits == digit)[0][0]
    ax[digit].imshow(images[idx], cmap='gray_r')
    ax[digit].set_title(str(digit))
    ax[digit].axis('off')

fig.savefig("outputs/sample_digits.png", bbox_inches='tight', dpi=150)
plt.close(fig)

# PCA Question 2:
print("\n")
print("-" * 100)
print("PCA: Question 2")
print("-" * 100)

pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(scores[:, 0], scores[:, 1],
                     c=y_digits, cmap='tab10', s=10)
plt.colorbar(scatter, label='Digit')
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA 2D Projection")
fig.savefig("outputs/pca_2d_projection.png")

plt.close(fig)

# PCA Question 3:
print("\n")
print("-" * 100)
print("PCA: Question 3")
print("-" * 100)

cum_var = np.cumsum(pca.explained_variance_ratio_)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(cum_var) + 1), cum_var, marker='.')
ax.set_xlabel("number of components")
ax.set_ylabel("cumulative explained variance")
ax.set_title("pca_variance_explained")
fig.savefig("outputs/pca_variance_explained.png")
plt.close(fig)

# Approx 12-15 components are needed to explain 80% of the variance

# PCA Question 4:
print("\n")
print("-" * 100)
print("PCA: Question 4")
print("-" * 100)


def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + \
            scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)


n_values = [2, 5, 15, 40]
sample_indices = list(range(5))

n_rows = len(n_values) + 1
n_nols = len(sample_indices)

fig, ax = plt.subplots(n_rows, n_nols, figsize=(8, 2*n_rows))

# Original row
for j, idx in enumerate(sample_indices):
    ax[0, j].imshow(images[idx], cmap='gray_r')
    ax[0, j].axis('off')
    if j == 0:
        ax[0, j].set_ylabel("Original")

# Recontruction rows:
for i, n in enumerate(n_values):
    for j, idx in enumerate(sample_indices):
        recon = reconstruct_digit(idx, scores, pca, n)
        ax[i+1, j].imshow(recon, cmap='gray_r')
        ax[i+1, j].axis('off')
        if j == 0:
            ax[i+1, j].set_ylabel(f"n = {n}")

fig.suptitle("PCA Recontrution")
fig.savefig("outputs/pca_reconstructions.png")
plt.close(fig)

# Digits become clearly recognizable around n=15, which roughly matches where
# the cumulative variance curve begins to level off (around 80% explained).
# At n=2 the images are very blurry; at n=5 you can start to see digit shapes
# but details are missing; by n=40 they are nearly indistinguishable from originals.
