from openai import OpenAI
import json
import os

os.system('cls')

# =====================================================================================
#                    Week 10 - Part 1: Warmup Exercise
# =====================================================================================

print('='*100)
print("=> Part 1: Warmup Exercise")
print('='*100)

# =====================================================================================
# LLMs as Transform
# =====================================================================================

print("=> LLMs as Transform")
print('='*100)

# =====================================================================================
# LLMs as Transform Question 1
# =====================================================================================

print("=> LLMs as Transform Question 1")
print('-'*100)

# Q1: For each task below, write a one-sentence comment saying whether you would use
# an LLM or deterministic code, and why.

# Task 1: Parse the string "Jan 5th, 2024" into an ISO date format like "2024-01-05".
# Answer:
# Use deterministic code. Date parsing is a well-defined, structured task with
# specific formats that can be reliably handled by regex or date parsing libraries.

# Task 2: Classify a customer support ticket -- "my card was charged twice" -- into one of:
# billing, technical, or general.
# Answer:
# Use an LLM. This requires understanding natural language nuance and context to
# classify unstructured customer input, which LLMs are well-suited for.

# Task 3: Calculate the average of a list of numbers.
# Answer:
# Use deterministic code. This is a simple mathematical operation with a clear,
# unambiguous definition that requires no AI.

# Task 4: Extract the company name from a freeform job title like
# "Sr. Data Eng @ Acme Corp (contract)".
# Answer:
# Use an LLM. Job titles vary widely in format and structure, and an LLM can
# robustly extract company names from freeform text better than brittle pattern matching.

# Task 5: Determine whether a product review is more than 100 words long.
# Answer:
# Use deterministic code. This is a simple string operation (count words and
# compare to a threshold) with no ambiguity or need for semantic understanding.

print("Comment for LLMs as Transform Question 1 added")

# =====================================================================================
# LLMs as Transform Question 2
# =====================================================================================

print('-'*100)
print("=> LLMs as Transform Question 2")
print('-'*100)

# Question 2:
# Your colleague has written the following pipeline prompt:
# system = "Summarize this product review in a few sentences."
#
# Problem this creates downstream in a pipeline:
# This prompt produces unstructured, variable-length text that is difficult to parse
# and store reliably. Different LLM calls may summarize differently (different lengths,
# formats, or completeness), making it hard to extract structured fields from the output
# for storage in databases or downstream processing. There's no guaranteed format.
#
# Improved prompt:
# system = """You are a review summarizer. Summarize the provided product review in
# exactly 2-3 sentences. Return your response in JSON format with the following
# structure: {"summary": "your summary here", "word_count": <number>}.
# Ensure the output is valid JSON that can be parsed programmatically."""

print("Comment for LLMs as Transform Question 2 added")

# =====================================================================================
# LLMs as Transform Question 3
# =====================================================================================

print('-'*100)
print("=> LLMs as Transform Question 3")
print('-'*100)

# Q3: Your dataset has 50,000 records and you need to run a classification call for each one
# using gpt-4o-mini. Answer:
#
# 1. If each call takes 1 second on average, how long would sequential processing take?
# Answer:
# 50,000 records × 1 second per call = 50,000 seconds ≈ 13.89 hours
#
# 2. What is one practical strategy to handle this more efficiently at scale, without
#    changing models?
# Answer:
# Use batch processing / parallel requests. Instead of processing records one at
# a time sequentially, make multiple concurrent API calls (respecting rate limits). For
# example, with 10 concurrent requests, you could reduce processing time to ~1.4 hours.
# Alternatively, use the OpenAI Batch API (if available) which offers lower costs and
# processes jobs asynchronously in parallel.

print("Comment for LLMs as Transform Question 3 added")

# =====================================================================================
# Azure OpenAI
# =====================================================================================

print('='*100)
print("=> LLMs as Transform")
print('='*100)

# =====================================================================================
# AAzure OpenAI Question 1
# =====================================================================================

print("Azure OpenAI Question 1")
print('-'*100)

# Q1: Name two reasons an organization might use Azure OpenAI instead of calling the
# OpenAI API directly. Be specific.
#
# Answer:
# 1. Data Residency & Compliance: Organizations can ensure their data stays within
#    specific geographic regions (e.g., EU, US) and comply with data sovereignty
#    regulations (GDPR, HIPAA, etc.). Azure OpenAI allows deployment in customer's
#    Azure subscriptions rather than sharing data with OpenAI's public infrastructure.
#
# 2. Virtual Network Integration & Security: Organizations can deploy Azure OpenAI
#    within a private Virtual Network (VNet) with network security groups, private
#    endpoints, and no public internet exposure. This provides end-to-end encryption
#    and isolates the service from the public internet, critical for enterprise security
#    requirements.

print("Comment for Azure OpenAI Question 1 added")

# =====================================================================================
# Azure OpenAI Question 2
# =====================================================================================

print('-'*100)
print("=> Azure OpenAI Question 2")
print('-'*100)

# Q2: When you switch from OpenAI to AzureOpenAI, the client initialization takes three
# Azure-specific parameters. Name them and describe what each one is (excluding api_key).
#
# Answer:
# 1. api_version: The API version string (e.g., "2024-02-15-preview") that specifies
#    which Azure OpenAI API version to use. Different versions may have different
#    capabilities and behaviors.
#
# 2. azure_endpoint: The base URL of your Azure OpenAI resource
#    (e.g., "https://my-resource.openai.azure.com"). This tells the client where your
#    Azure OpenAI deployment is located.
#
# 3. azure_deployment (used in chat.completions.create()): The name of the specific
#    deployment within your Azure OpenAI resource (e.g., "my-gpt-4o-mini-deployment").
#    Azure allows multiple model deployments per resource, so you specify which
#    deployment to use.

print("Comment for Azure OpenAI Question 2 added")

# =====================================================================================
# Azure OpenAI Question 3
# =====================================================================================

print('-'*100)
print("=> Azure OpenAI Question 3")
print('-'*100)

# Q3: When using AzureOpenAI, the model parameter in chat.completions.create() does not
# take a value like "gpt-4o-mini". What does it take instead, and where do you find the
# right value to use?
#
# Answer:
# The model parameter should be set to the azure_deployment name, NOT the model name
# (e.g., "my-gpt-4o-mini-deployment" instead of "gpt-4o-mini"). You find the correct
# value in the Azure Portal: navigate to your Azure OpenAI resource, go to the
# "Model deployments" section, and copy the deployment name you created. The deployment
# name is what you chose when you deployed a specific model to Azure OpenAI.

print("Comment for Azure OpenAI Question 3 added")
