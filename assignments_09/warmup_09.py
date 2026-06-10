import os

os.system('cls')

# =====================================================================================
#                    Week 9 - Part 1: Warmup Exercise
# =====================================================================================

print('='*100)
print("Part 1: Warmup Exercise")
print('='*100)

# =====================================================================================
# Azure Authentication
# =====================================================================================

print("---------------------- Azure Authentication ----------------------")
print('='*100)

# =====================================================================================
# Azure Authentication Question 1
# =====================================================================================

print("=> Azure Authentication Question 1")
print('-'*100)
# Question:
# when you run a Python script locally that uses DefaultAzureCredential, what does it
# rely on to authenticate? What command must you have run first, and how does
# DefaultAzureCredential know to use it?

# Asnwer:
# When you run a Python script locally that uses DefaultAzureCredential, it relies on
# the cached token from the Azure CLI. You must have run `az login` first, which opens
# a browser for interactive authentication and stores a refresh token on your machine.

# DefaultAzureCredential tries a chain of credential sources in order. Locally, it finds
# the AzureCliCredential in that chain, which reads the token that `az login` saved.
# So you don't pass any secrets in code — it just picks up whatever session is already
# active from the CLI.

print("Comment added for Question 1")

# =====================================================================================
# Azure Authentication Question 2
# =====================================================================================

print('-'*100)
print("=> Azure Authentication Question 2")
print('-'*100)

# Question:
# why can't a deployed pipeline (running on an Azure VM or container) use az login for
# authentication? What does it use instead, and why does the same Python code work
# without changes?

# Answer:
# A deployed pipeline running on an Azure VM or container can't use `az login` because
# there's no human present to complete the interactive browser login. It also wouldn't
# make sense to store personal credentials on shared infrastructure.

# Instead, it uses a Managed Identity — a system-assigned or user-assigned identity that
# Azure automatically provisions for the resource. The VM or container can request tokens
# from a local metadata endpoint without any stored secrets.

# The same Python code works without changes because DefaultAzureCredential's credential
# chain includes ManagedIdentityCredential. When it runs on Azure infrastructure, it
# detects the managed identity endpoint and authenticates through that, so no code change
# is needed — just a different environment.

print("Comment added for Question 2")

# =====================================================================================
# Azure Authentication Question 3
# =====================================================================================

print('-'*100)
print("=> Azure Authentication Question 3")
print('-'*100)

# Question:
# You run a script that creates a DefaultAzureCredential and immediately gets an
# AuthenticationError. In a comment block, describe the two most likely causes and
# how you would diagnose each.

# Answer:
# Two most likely causes of an immediate AuthenticationError with DefaultAzureCredential:

# 1) You never ran `az login`, or your session expired.
#    Diagnosis: Run `az account show` in the terminal. If it errors or shows an expired
#    token, run `az login` again to refresh the session.

# 2) The azure-identity package can't find any valid credential in its chain (e.g.,
#    no environment variables set, no managed identity available, no CLI session).
#    Diagnosis: Enable logging with `logging.basicConfig(level=logging.DEBUG)` or pass
#    `logging_enable=True` to the credential. The output will show each credential type
#    it tried and why each one failed, which tells you exactly what's missing.

print("Comment added for Question 3")

# =====================================================================================
# Blob Storage
# =====================================================================================

print('='*100)
print("---------------------- Blob Storage ----------------------")
print('='*100)

# =====================================================================================
# Blob Storage Question 1
# =====================================================================================

print("=> Blob Storage Question 1")
print('-'*100)

# Question:
# describe the three-level hierarchy of Azure Blob Storage in your own words. Give a concrete
# analogy that maps each level to something familiar (a filesystem, a filing cabinet, etc.)

# Answer:
# Azure Blob Storage has a three-level hierarchy:
#
# 1) Storage Account — the top-level namespace. Think of it like the building that
#    houses all your filing cabinets. It defines the access policies, redundancy settings,
#    and the unique URL endpoint (e.g., https://<account>.blob.core.windows.net).
#
# 2) Container — a logical grouping of blobs inside the account. This is like one
#    filing cabinet drawer: it holds related files together and you can set permissions
#    at this level. Example: "raw-data" container vs. "processed-data" container.
#
# 3) Blob — the actual file/object stored inside a container. This is the individual
#    document in the drawer. It can be any type: CSV, JSON, image, parquet, etc.
#
# Filesystem analogy:
#   Storage Account  →  Drive (e.g., C:\)
#   Container        →  Top-level folder (e.g., C:\raw-data\)
#   Blob             →  File (e.g., C:\raw-data\2024-01-15.json)

print("Comment added for Question 1")

# =====================================================================================
# Blob Storage Question 2
# =====================================================================================

print('-'*100)
print("=> Blob Storage Question 2")
print('-'*100)

# Question:
# For each scenario below, write one sentence in a comment block saying whether you would
# use Blob Storage or a relational database (like Azure SQL), and why.

# 1. A REST API returns a JSON payload each hour. You need to store the raw responses for reprocessing later.
# 2. Your pipeline produces a table of 50 million customer transactions that your analytics team queries by date range and customer ID every day.
# 3. A computer vision model produces image embeddings as NumPy arrays. You need to save them between pipeline runs.

# Answer:
# Scenario 1: A REST API returns a JSON payload each hour; store raw responses for reprocessing.
# → Blob Storage, because these are raw immutable files that just need to be saved and
#   read back whole — no querying by columns or joining to other tables is needed.

# Scenario 2: 50 million customer transactions queried by date range and customer ID daily.
# → Relational database (Azure SQL), because the analytics team needs indexed, structured
#   queries with filters and aggregations — exactly what SQL is designed for.

# Scenario 3: Image embeddings as NumPy arrays saved between pipeline runs.
# → Blob Storage, because these are binary array files (e.g., .npy) that get written once
#   and loaded whole into memory — there's no need for row-level querying.

print("Comment added for Question 2")

# =====================================================================================
# Blob Storage Question 3
# =====================================================================================

print('-'*100)
print("=> Blob Storage Question 3")
print('-'*100)


def list_container(container_client):
    """Print the name and size (in bytes) of every blob in the container."""
    for blob in container_client.list_blobs():
        print(f"{blob.name} - {blob.size} bytes")


print("Function list_container() defined")

# =====================================================================================
# Blob Storage Question 4
# =====================================================================================

print('-'*100)
print("=> Blob Storage Question 4")
print('-'*100)


def upload_text(container_client, blob_name, text):
    """Encode a string as UTF-8 and upload it as a blob, overwriting if it exists."""
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(text.encode("utf-8"), overwrite=True)


print("Function upload_text() defined")
