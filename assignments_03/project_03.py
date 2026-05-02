import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.inspection import DecisionBoundaryDisplay

warnings.filterwarnings("ignore", category=RuntimeWarning)

os.system('cls')

os.makedirs("outputs", exist_ok=True)

# =====================================================================================
#                    Mini Project -- Spam or Ham - A Classifier Shootout
# =====================================================================================

# ------------------------------------------------------------------------------------
#  Task 1: Load and Explore
# ------------------------------------------------------------------------------------

print("-" * 100)
print("Task 1: Load and Explore")
print("-" * 100)

# Col names:
COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"

df = pd.read_csv(url, header=None, names=COLUMN_NAMES)

print(f"Dataset Shape: {df.shape}")
print(f"Class Distribution:\n{df['spam_label'].value_counts()}")
print(f"\nClass Propostions:\n{df['spam_label'].value_counts(normalize=True)}")

# The dataset has about 60% HAM and 40% SPAM, which is balanced
# With this balance, war accuracy is a reasonable
features_to_plot = ["word_freq_free",
                    "char_freq_!", "capital_run_length_total"]
for feat in features_to_plot:
    fig, ax = plt.subplots(figsize=(6, 4))
    df.boxplot(column=feat, by="spam_label", ax=ax)
    ax.set_xlabel("Spam_label (Ham = 0, SPAM = 1)")
    ax.set_ylabel(feat)
    ax.set_title(f"{feat} by Class")
    fig.suptitle("")
    plt.savefig(f"outputs/{feat}.png", dpi=350, bbox_inches="tight")

# Are the differences between classes dramatic or subtle?
# 1. word_freq_free - near to 0 for ham but has many high outliers for spam
# 2. char_freq_! - much higher in spam mails
# 3. Capital_run_length_total -Large in Spam


# What does this heavy skew toward zero tell you about the data?
# Why does the numeric scale vary so dramatically across features (some are tiny fractions, others reach into the thousands)?
# Why might that matter for some of the models you are about to build?

# Many word-frequency features are zero for most emails because a typical email
# doesn't contain words like "free" or "money". This creates heavy right skew.
# The numeric scales vary wildly -- word frequencies are small fractions (0-100%
# of words) while capital_run_length_total can be in the thousands. This matters
# for distance-based models like KNN and coefficient-based models like logistic
# regression, since features with larger magnitudes will dominate unless scaled.

# ------------------------------------------------------------------------------------
#  Task 2: Prepare Your Data
# ------------------------------------------------------------------------------------

print("-" * 100)
print("Task 2: Prepare Your Data")
print("-" * 100)

# train/test split
X = df.drop(columns=["spam_label"])
y = df["spam_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")

# scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA preprocessing
pca = PCA()
pca.fit(X_train_scaled)

# Plot the cumulative explained variance
cum_var = np.cumsum(pca.explained_variance_ratio_)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(cum_var) + 1), cum_var, marker='.')
ax.set_xlabel("Number of components")
ax.set_ylabel("cumulative explained variance")
ax.set_title("PCA cumulative explained variance")
fig.savefig("outputs/pca_cumulative_explained_variance.png",
            bbox_inches='tight', dpi=150)
plt.close(fig)


n = int(np.argmax(cum_var >= 0.9) + 1)
print(f"Number of compoenets for 90% variance: {n}")

# PCA-reduced arrays
X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca = pca.transform(X_test_scaled)[:, :n]
print(
    f"PCA Reduced shape: train-{X_train_pca.shape}, test-{X_test_pca.shape}")


# ------------------------------------------------------------------------------------
#  Task 3: A Classifier Comparison
# ------------------------------------------------------------------------------------

print("-" * 100)
print("Task 3: A Classifier Comparison")
print("-" * 100)

models = {}

# 1. knn on unscaled data
print(f"1. KNN on unscaled data ----->")
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
y_pred = knn_unscaled.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
models["KNN (unscaled)"] = ("unscaled", knn_unscaled)

# 2a. knn on scaled data
print(f"2a. KNN on scaled data ----->")
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_ks = knn_scaled.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred_ks):.4f}")
print(classification_report(y_test, y_pred_ks, target_names=["ham", "spam"]))
models["KNN (scaled)"] = ("scaled", knn_scaled)

# 2b. knn on PCA-reduced data
print(f"2b. KNN on PCA-reduced data ----->")
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
y_pred_kp = knn_pca.predict(X_test_pca)
print(f"Accuracy: {accuracy_score(y_test, y_pred_kp):.4f}")
print(classification_report(y_test, y_pred_kp, target_names=["ham", "spam"]))
models["KNN (PCA)"] = ("pca", knn_pca)

# 3.Decision tree
print(f"3. Decision tree ----->")
for depth in [3, 5, 10, None]:
    dt = DecisionTreeClassifier(random_state=42, max_depth=depth)
    dt.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, dt.predict(X_train))
    test_acc = accuracy_score(y_test, dt.predict(X_test))
    print(
        f"max_depth={str(depth):>4s} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

# What do you notice as depth increases? What does that tell you about overfitting?
# As depth increases, training accuracy rises toward 1.0 but test accuracy may
# plateau or drop -- classic overfitting.  depth=None memorizes the training data
# (train_acc=1.0) but generalizes worse.  I would use max_depth=10 in production:
# it captures enough complexity without severe overfitting

choosen_depth = 10

dt_final = DecisionTreeClassifier(max_depth=choosen_depth, random_state=42)
dt_final.fit(X_train, y_train)
y_pred_det = dt_final.predict(X_test)
print(f"Decision Tress (MAX Depeth = {choosen_depth})")
print(f"Accuracy: {accuracy_score(y_test, y_pred_det):.4f}")
print(classification_report(y_test, y_pred_det, target_names=["ham", "spam"]))
models["Decision Tree"] = ("unscaled", dt_final)

# 4. Random Forest cloassifier
print("4. Random Forest classifier ----->")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf, target_names=["ham", "spam"]))
models["Random Forest Class"] = ("unscaled", rf)


# 5a. Logistic Regression on Scaled Data
print("5a. Logistic Regression on Scaled Data ----->")
lr_scaled = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
lr_scaled.fit(X_train_scaled, y_train)
y_pred_lr = lr_scaled.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(classification_report(y_test, y_pred_lr, target_names=["ham", "spam"]))
models["Logistic Regression (scaled)"] = ("scaled", lr_scaled)


# 5b. Logistic Regression on PCA-reduced data
print("5b. Logistic Regression on PCA-reduced data ----->")
lr_pca = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
lr_pca.fit(X_train_pca, y_train)
y_pred_lrp = lr_pca.predict(X_test_pca)
print(f"Accuracy: {accuracy_score(y_test, y_pred_lrp):.4f}")
print(classification_report(y_test, y_pred_lrp, target_names=["ham", "spam"]))
models["Logistic Regression (PCA)"] = ("pca", lr_pca)


# Summary:
# Random Forest works the best here. KNN (scaled) and Logistic Regression
# (scaled) are close behind.
# Scaling matters a lot for KNN -- without it, big features like
# capital_run_length_total take over and smaller features get ignored.
# PCA can hurt KNN and Logistic Regression a little because it throws away
# some useful info when it reduces the number of features.
# For a spam filter, it's more important to avoid marking real emails as spam
# than to catch every spam email. Losing a real email is a bigger problem
# than letting some spam through. So we want high precision on spam, even if
# we miss a few spam emails.


# Feature importances -- Decision Tree vs Random Forest
print("Feature Importances ----->")
feature_names = X.columns.tolist()

dt_importances = dt_final.feature_importances_
rf_importances = rf.feature_importances_

dt_top10_idx = np.argsort(dt_importances)[::-1][:10]
rf_top10_idx = np.argsort(rf_importances)[::-1][:10]

print("\nTop 10 features (Decision Tree):")
for i in dt_top10_idx:
    print(f" {feature_names[i]:35s}  {dt_importances[i]:.4f}")

print("\nTop 10 features (Random Forest):")
for i in rf_top10_idx:
    print(f" {feature_names[i]:35s} {rf_importances[i]:.4f}")

# Both models pick similar top features -- things like char_freq_!,
# word_freq_free, capital_run_length_total, and char_freq_$ show up near
# the top in both. This makes sense because spam emails tend to have more
# exclamation marks, the word "free", ALL CAPS text, and dollar signs
# compared to normal emails.

# Bar chart of Random Forest feature importances
sorted_idx = np.argsort(rf_importances)[::-1][:15]  # top 15 for readability
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(sorted_idx)), rf_importances[sorted_idx][::-1])
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels([feature_names[i] for i in sorted_idx][::-1])
ax.set_xlabel("Feature Importance")
ax.set_title("Random Forest -- Top 15 Feature Importances")
fig.savefig("outputs/feature_importances.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("Saved outputs/feature_importances.png")

# Confusion matrix for best model
# Determine best model by test accuracy
best_name = None
best_acc = 0
best_preds = None
for name, (dtype, model) in models.items():
    if dtype == "unscaled":
        preds = model.predict(X_test)
    elif dtype == "scaled":
        preds = model.predict(X_test_scaled)
    else:
        preds = model.predict(X_test_pca)
    acc = accuracy_score(y_test, preds)
    if acc > best_acc:
        best_acc = acc
        best_name = name
        best_preds = preds

print(f"\nBest model: {best_name} (accuracy: {best_acc:.4f})")
cm = confusion_matrix(y_test, best_preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["ham", "spam"])
fig, ax = plt.subplots()
disp.plot(ax=ax)
ax.set_title(f"Confusion Matrix -- {best_name}")
fig.savefig("outputs/best_model_confusion_matrix.png",
            bbox_inches="tight", dpi=150)
plt.close(fig)
print("Saved outputs/best_model_confusion_matrix.png")

# The best model probably lets some spam slip through rather than accidentally
# blocking real emails. This is the better mistake to make -- missing a spam
# email is no big deal, but losing an important real email is a real problem.

# ------------------------------------------------------------------------------------
# Task 4: Cross-Validation
# ------------------------------------------------------------------------------------

print("-" * 100)
print("Task 4: Cross-Validation ----->")
print("-" * 100)

cv_configs = {
    "KNN (unscaled)":                   (KNeighborsClassifier(n_neighbors=5), X_train),
    "KNN (scaled)":                     (KNeighborsClassifier(n_neighbors=5), X_train_scaled),
    "KNN (PCA)":                        (KNeighborsClassifier(n_neighbors=5), X_train_pca),
    "Decision Tree":                    (DecisionTreeClassifier(max_depth=choosen_depth, random_state=42), X_train),
    "Random Forest":                    (RandomForestClassifier(n_estimators=100, random_state=42), X_train),
    "Logistic Regression (scaled)":     (LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'), X_train_scaled),
    "Logistic Regression (PCA)":        (LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'), X_train_pca),
}

print(f"\n{'Model':40s}  {'Mean':>6s}  {'Std':>6s}")
print("-" * 56)
for name, (model, data) in cv_configs.items():
    scores = cross_val_score(model, data, y_train, cv=5)
    print(f"{name:40s}  {scores.mean():.4f}  {scores.std():.4f}")

# Random Forest usually scores the highest and gives the most consistent results
# across folds because it averages 100 trees, which evens things out. The
# Decision Tree jumps around more between folds since a single tree is more
# affected by which data it sees. The overall ranking should be similar to
# what we saw in Task 3.

# ------------------------------------------------------------------------------------
# Task 5: Building a Prediction Pipeline
# ------------------------------------------------------------------------------------
print('-'*100)
print("Task 5: Building a Prediction Pipeline")
print('-'*100)

# Pipeline 1: Best tree-based model (Random Forest)
# Tree-based models don't need scaling or PCA, so this pipeline is just the model.
tree_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])
tree_pipeline.fit(X_train, y_train)
y_pred_tree_pipe = tree_pipeline.predict(X_test)
print("Pipeline: Random Forest ----->")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree_pipe):.4f}")
print(classification_report(
    y_test, y_pred_tree_pipe, target_names=["ham", "spam"]))

# Pipeline 2: Best non-tree model (Logistic Regression with scaling)
# We always scale the data since Logistic Regression needs it. We only add
# PCA if it actually helped in Task 3; otherwise we skip it.

acc_lr_scaled = accuracy_score(y_test, y_pred_lr)
acc_lr_pca = accuracy_score(y_test, y_pred_lrp)

if acc_lr_pca > acc_lr_scaled:
    # PCA helped -- include it in the pipeline
    nontree_pipeline = Pipeline([
        ("scaler",          StandardScaler()),
        ("pca",             PCA(n_components=n)),
        ("classifier",      LogisticRegression(
            C=1.0, max_iter=1000, solver='liblinear'))
    ])
    pipe_label = "Logistic Regression (scaler + PCA)"
else:
    # PCA didn't help -- just scale
    nontree_pipeline = Pipeline([
        ("scaler",      StandardScaler()),
        ("classifier",  LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
    ])
    pipe_label = "Logistic Regression (scaler only)"

nontree_pipeline.fit(X_train, y_train)
y_pred_nontree_pipe = nontree_pipeline.predict(X_test)
print(f"Pipeline: {pipe_label} ----->")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nontree_pipe):.4f}")
print(classification_report(
    y_test, y_pred_nontree_pipe, target_names=["ham", "spam"]))

# The two pipelines look different because tree models don't care about
# feature scales, but Logistic Regression does -- so it needs a scaler
# (and maybe PCA) built in.
#
# Why use a pipeline?
# 1. It runs all the steps in the right order automatically.
# 2. It keeps the scaler/PCA from peeking at test data (no data leakage).
# 3. You get one single object to save and use later -- just call predict().
# 4. Easy to hand off to someone else -- they don't need to know the details.

print("\nAll tasks complete.")
