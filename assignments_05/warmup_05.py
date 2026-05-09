from dotenv import load_dotenv
from openai import OpenAI
import json

import os
os.system('cls')

# ------------------------------------------------------------------------------------------------------
# Week 5 - Part 1: Warmup Exercises
# ------------------------------------------------------------------------------------------------------

print('-'*100)
print("                                 Week 5 - Part 1: Warmup Exercises")

# ------------------------------------------------------------------------------------------------------
# The Chat Completions API
# ------------------------------------------------------------------------------------------------------

print('-'*100)
print(" 1. The Chat Completions API")

# ------------------------------------------------------------------------------------------------------
# API Question 1
# ------------------------------------------------------------------------------------------------------

print('-'*100)
print(" API Question 1")
print('-'*100)

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user",
               "content": "What is one thing that makes Python a good language for beginners?"}]
)
print("Response:\n", response.choices[0].message.content)
print(f"Model: {response.model}")
print(f"Total Tokens Used: {response.usage.total_tokens}")

# ------------------------------------------------------------------------------------------------------
# API Question 2
# ------------------------------------------------------------------------------------------------------

print('-'*100)
print(" API Question 2")
print('-'*100)

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    print(f"Temparature {temp}: {response.choices[0].message.content}")

# Temparature 0:
# It produces the most consistent output - same in each run

# Temparature 0.7:
# It adds moderate creativity and variation in runs

# Temparature 1.5:
# It produces highly creative and descriptive but high variations

# Observation:
# For consistent, reproducible output, use temparature=0.

# ------------------------------------------------------------------------------------------------------
# API Question 3
# ------------------------------------------------------------------------------------------------------

print('-'*100)
print(" API Question 3")
print('-'*100)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user",
               "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

for i, choice in enumerate(response.choices):
    print(f"Response {i+1}: {choice.message.content}\n")

# ------------------------------------------------------------------------------------------------------
# API Question 4
# ------------------------------------------------------------------------------------------------------

print('-'*100)
print(" API Question 4")
print('-'*100)

prompt = 'Explain how neural networks work.'
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=15
)

print(f"Repsonse: {response.choices[0].message.content}")

# ------------------------------------------------------------------------------------------------------
# System Messages and Personas
# ------------------------------------------------------------------------------------------------------

print('-'*100)
print(" 2. System Messages and Personas")

# ------------------------------------------------------------------------------------------------------
# System Question 1
# ------------------------------------------------------------------------------------------------------

print('-'*100)
print(" System Question 1")
print('-'*100)

# Personality 1:
message_1 = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=message_1
)

print(f"Tutor Repsonse: {response.choices[0].message.content}")
print(f"Total Tokens Used: {response.usage.total_tokens}")

# Personality 2:
message_2 = [
    {"role": "system", "content": "You are a sarcastic stand-up comedian who reluctantly answers programming questions with jokes and dry humor."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=message_2
)

print(f"Comedy Repsonse: {response.choices[0].message.content}")
print(f"Total Tokens Used: {response.usage.total_tokens}")

# The system message completely changes the tone and style of the response.
# The tutor gives a gentle, clear explanation with encouragement, while the comedian
# delivers the same information wrapped in sarcasm and humor. The factual content
# is similar, but the delivery is drastically different.

# ------------------------------------------------------------------------------------------------------
# System Question 1
# ------------------------------------------------------------------------------------------------------

print('-'*100)
print(" System Question 1")
print('-'*100)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print(f"Repsonse: {response.choices[0].message.content}")

# Message knows Jordan's name because we passed that in the message.
# The API is stateless - it has no memory. But by including user and message, we gives a model
# all data it need to "REMEMBER" earlier parts of conversation.

# ------------------------------------------------------------------------------------------------------
# Prompt Engineering
# ------------------------------------------------------------------------------------------------------

print('-'*100)
print(" 3. Prompt Engineering")

# ------------------------------------------------------------------------------------------------------
# Prompt Question 1 — Zero-Shot
# ------------------------------------------------------------------------------------------------------

print('-'*100)
print(" Prompt Question 1 — Zero-Shot")
print('-'*100)

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews, 1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Classify the sentiment of each review below as positive, negative, or mixed."},
            {"role": "user", "content": review}
        ]
    )

print(f"Review {i}: {response.choices[0].message.content}")

# ------------------------------------------------------------------------------------------------------
# Prompt Question 2 — One-Shot
# ------------------------------------------------------------------------------------------------------

print('-'*100)
print(" Prompt Question 2 — One-Shot")
print('-'*100)

for i, review in enumerate(reviews, 1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Classify the sentiment of a review as potsitive, negative or mixed"},
            {"role": "user", "content": 'Review: "Fast shipping but the item arrived damaged."'},
            {"role": "assistant", "content": "Sentiment: mixed"},
            {"role": "user", "content": f'Review: "{review}"'}
        ]
    )
    print(f"Review {i}: {response.choices[0].message.content}")

# Adding example (shot) improves  the consistency of the output
# Model follows the "Sentiment: mixed" format shown in example.
# whereas zero-shot responses may vary in format depending on how the model interprets the task.

# ------------------------------------------------------------------------------------------------------
# Prompt Question 3 — Few-Shot
# ------------------------------------------------------------------------------------------------------

print('-'*100)
print(" Prompt Question 3 — Few-Shot")
print('-'*100)

for i, review in enumerate(reviews, 1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Classify the sentiment of a review as positive, negative, or mixed."},
            {"role": "user", "content": 'Review: "The staff was friendly and the food was delicious."'},
            {"role": "assistant", "content": "Sentiment: positive"},
            {"role": "user", "content": 'Review: "Terrible experience, the product broke after one day."'},
            {"role": "assistant", "content": "Sentiment: negative"},
            {"role": "user", "content": 'Review: "Fast shipping but the item arrived damaged."'},
            {"role": "assistant", "content": "Sentiment: mixed"},
            {"role": "user", "content": f'Review: "{review}"'}
        ]
    )
    print(f"Review {i}: {response.choices[0].message.content}")

# Zero-shot: Quick and simple, best when the task is straightforward and format doesn't matter much.
# One-shot: Adds format guidance with minimal extra tokens, good when you need consistent output format.
# Few-shot: Most reliable for consistent format and accurate classification, especially for nuanced tasks.
# Use zero-shot for simple/clear tasks, one-shot when format matters, and few-shot when accuracy
# and consistency are critical or the task is ambiguous.

# ------------------------------------------------------------------------------------------------------
# Prompt Question 4 — Chain of Thought
# ------------------------------------------------------------------------------------------------------

print('-'*100)
print(" Prompt Question 4 — Chain of Thought")
print('-'*100)

system_promt = "You are helpful assistant. Show your reasoining step by step before giving a final answer. Label the final answer clearly."

user_promt = 'A data engineer earns $85,000 per year. She gets a 12% raise, \
then 6 months later takes a new job that pays $7,500 more per year than her post-raise salary. \
What is her final annual salary?'

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_promt},
        {"role": "user", "content": user_promt}
    ]
)
print(f"Response: {response.choices[0].message.content}")

# Asking the model to reason step by step (chain of thought) tends to improve accuracy because
# it forces the model to break the problem into smaller parts and solve each one before combining.
# Without step-by-step reasoning, the model may try to jump to a final answer and make arithmetic
# or logical errors along the way.

# ------------------------------------------------------------------------------------------------------
# Prompt Question 5 —  Structured Output
# ------------------------------------------------------------------------------------------------------

print('-'*100)
print(" Prompt Question 5 —  Structured Output")
print('-'*100)

review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

prompt = "Analyze the sentiment of the following review. Return your result only as valid JSON  \
with these keys: sentiment (positive, negative, or mixed), confidence (a float from 0 to 1), \
and reason (one sentence). Do not include any text outside the JSON."

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": review}
    ]
)
raw_response = response.choices[0].message.content
# print(f"Raw Response: {raw_response}")

# Strip markdown code fences if present
cleaned = raw_response.strip()
if cleaned.startswith("```"):
    cleaned = cleaned.split("\n", 1)[1]  # remove first line (```json)
    cleaned = cleaned.rsplit("```", 1)[0]  # remove closing ```
    cleaned = cleaned.strip()

try:
    parsed = json.loads(cleaned)
    print(f"Sentiment: {parsed['sentiment']}")
    print(f"Confidence: {parsed['confidence']}")
    print(f"Reason: {parsed['reason']}")
except json.JSONDecodeError:
    print(f"Failed to Parse JSON Raw Response")
    print(raw_response)

# ------------------------------------------------------------------------------------------------------
# Prompt Question 6 —  Delimiters
# ------------------------------------------------------------------------------------------------------

print('-'*100)
print(" Prompt Question 6 — Delimiters")
print('-'*100)

# Test 1:
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)
print(f"Instrcutions Text: {response.choices[0].message.content}")

# Test 2:
user_text_2 = "The weather was sunny and warm. I spend afternoon outside."

prompt_2 = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text_2}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt_2}
    ]
)
print(f"Non-Instrcutions Text: {response.choices[0].message.content}")

# ------------------------------------------------------------------------------------------------------
# Local Models with Ollama
# ------------------------------------------------------------------------------------------------------

print('-'*100)
print(" 4. Local Models with Ollama")

# ------------------------------------------------------------------------------------------------------
# Ollama Question
# ------------------------------------------------------------------------------------------------------

print('-'*100)
print(" Ollama Question ")
print('-'*100)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Explain what a large language model is in two sentences."}
    ]
)

print(f"OpenAI Response: {response.choices[0].message.content}")


# Ollama terminal output (paste your output from: ollama run qwen3:0.6b "Explain what a large language model is in two sentences."):
# """
# Thinking...
# Okay, the user wants me to explain a large language model in two sentences. Let me start by
# breaking down the key points. First, a large language model is a type of AI that can understand
# and generate text. I should mention that it's trained on vast amounts of text to make it accurate
# and helpful. Then, I need to highlight its capabilities, like being able to write coherent
# stories or answer questions. Wait, but the user asked for two sentences. Let me make sure each
# sentence covers a different aspect. The first sentence should define the model, the second can
# talk about its use cases. Let me check if I'm not missing anything. Oh, right, emphasizing that
# it's a powerful tool for various tasks. Alright, that should cover it.
# ...done thinking.

# A large language model is a type of AI that can understand and generate text, trained on vast amounts of
# information to make it accurate and helpful. It can write coherent stories, answer questions, or perform
# tasks like writing code, making it a powerful tool for various applications.
# """

# Differences: The OpenAI response (gpt-4o-mini) is typically more polished and concise, while
# the local Ollama model (qwen3:0.6b) may produce less refined or slightly less accurate output
# due to its much smaller size (0.6B vs hundreds of billions of parameters).
#
# Advantage of running locally: No API costs, full data privacy — nothing leaves your machine.
# Disadvantage of running locally: Smaller models are less capable, and larger local models
# require significant hardware (GPU/RAM) to run at reasonable speeds.
