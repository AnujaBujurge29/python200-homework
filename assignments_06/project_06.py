from dotenv import load_dotenv
import os
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# =====================================================================================
#                    Week 6 - Part 1: Mini Project -- Groundwork Coffee Co. Q&A Assistant
# =====================================================================================

print('='*100)
print("--------------- Week 6 - Part 1: Mini Project -- Groundwork Coffee Co. Q&A Assistant ---------------")
print('='*100)

# ------------------------------------------------------------------------------------
#  Step 1: Setup
# ------------------------------------------------------------------------------------

print("-" * 100)
print("Step 1: Setup")
print("-" * 100)

# Load your API key
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

docs_dir = Path(
    "../../python-200/lessons/06_AI_augmentation/resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

print(f"Directory verified: {docs_dir}")

# ------------------------------------------------------------------------------------
#  Step 2: Load the Documents
# ------------------------------------------------------------------------------------

print("-" * 100)
print("Step 2: Load the Documents")
print("-" * 100)

documents = SimpleDirectoryReader(str(docs_dir)).load_data()

print(f"Total Documents loaded: {len(documents)}")
for i, doc in enumerate(documents, 1):
    file_name = doc.metadata.get("file_name", "Unknown file")
    print(f" {i}. {file_name}")


# ------------------------------------------------------------------------------------
#  Step 3: Build the Index and Query Engine
# ------------------------------------------------------------------------------------

print("-" * 100)
print("Step 3: Build the Index and Query Engine")
print("-" * 100)

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)

print(f"Index build successfully")

# ------------------------------------------------------------------------------------
#  Step 4: Query the Assistant
# ------------------------------------------------------------------------------------

print("-" * 100)
print("Step 4: Query the Assistant")
print("-" * 100)

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for i, question in enumerate(questions, 1):
    print(f"\n{'='*70}")
    print(f"Q{i}: {question}")

    response = query_engine.query(question)
    print(f"\nAnswer:\n{response}")

    if response.source_nodes:
        top_node = response.source_nodes[0]
        doc_name = top_node.metadata.get("file_name", "Unknown")
        score = top_node.score if top_node.score is not None else "N/A"
        snippet = top_node.get_content()[:200].replace("\n", " ")
        print(f"\nTop source:")
        print(f" Document: {doc_name}")
        print(f" Similarity score: {score}")
        print(f" Snippet: {snippet}...")
    else:
        print("\n(No source nodes retrieved)")

# Reflection on the responses:
# After running all queries, consider:
# - Did the assistant sound confident in its answers, or did it hedge with phrases like "based on the provided information"?
# - Were the retrieved documents actually relevant to each question?
# - Did any answer surprise you — either because it was accurate or inaccurate?
# - Which queries seemed easiest vs. hardest for the RAG system?


# ------------------------------------------------------------------------------------
#  Step 5: Find a Failure
# ------------------------------------------------------------------------------------

print("-" * 100)
print("Step 5: Find a Failure")
print("-" * 100)

# Ask a question designed to be hard for the system.
hard_question = "What is Groundwork's environmental sustainability policy, and how does it compare to industry standards?"

print(f"\n{'='*70}")
print("STEP 5: Hard Question Test")
print(f"\nQuestion: {hard_question}")

hard_response = query_engine.query(hard_question)
print(f"\nAnswer:\n{hard_response}")

print("\nAll three retrieved source nodes:")
if hard_response.source_nodes:
    for j, node in enumerate(hard_response.source_nodes, 1):
        doc_name = node.metadata.get("file_name", "Unknown")
        score = node.score if node.score is not None else "N/A"
        snippet = node.get_content()[:200].replace("\n", " ")
        print(f"\n[{j}] {doc_name}")
        print(f"   Similarity score: {score}")
        print(f"   Text: {snippet}...")
else:
    print("(No source nodes retrieved)")

# Step 5 analysis:
# Why this question is hard: This query asks for an environmental sustainability policy, which may not
# exist in the Groundwork documents (they are likely focused on menu, hours, and loyalty program info).
# It also asks for a comparison to industry standards, which definitely requires external knowledge.
#
# What went wrong: The retriever will likely pull loosely related chunks (maybe something about
# eco-friendly practices if mentioned in passing, or return general business info that is not relevant).
# The model may then either hedge ("I don't have information about...") or confabulate an answer based
# on generic knowledge.
#
# Tone observation: If the model sounds equally confident on hard questions as on easy ones, it is a
# red flag. A good RAG system (or honest model) should admit uncertainty when sources don't support
# the answer. If tone didn't change, that is a hallucination risk signal.
#
# Improvements: Add a confidence threshold so the system only responds when similarity scores are high.
# Implement an explicit "I don't know" response for low-confidence queries. Add document tags so the
# system can skip irrelevant documents upfront.

# ------------------------------------------------------------------------------------
#  Step 6: Reflection
# ------------------------------------------------------------------------------------

print("-" * 100)
print("Step 6: Reflection")
print("-" * 100)

# Step 6: Reflection on the RAG project.
#
# 1) Code efficiency — from lesson to framework:
# The lesson showed that building RAG manually requires: document loading, text splitting, embedding
# generation, vector index creation, similarity search, and response generation — likely 50+ lines of
# explicit code. This project achieved the same with ~10 lines (SimpleDirectoryReader, VectorStoreIndex,
# query_engine). This demonstrates the huge value of frameworks: they hide complexity, reduce bugs,
# and let you focus on the business logic rather than infrastructure. Using LlamaIndex cuts development
# time dramatically.
#
# 2) Real-world use cases beyond coffee shops:
# - Legal firms: Build an AI assistant that answers questions about internal case files, contracts,
#    and precedents — without lawyers having to manually search thousands of documents.
# - Healthcare: A patient education system that answers questions about a clinic's specific treatment
#    protocols, insurance policies, and patient guidelines without hallucinating medical advice.
# - Corporate HR: An employee handbook bot that answers policy questions consistently without making
#    up benefits or rules.
# - Research institutions: A system that helps researchers search and synthesize findings across
#    hundreds of papers on a specific topic.
#
# 3) A fundamental RAG failure mode:
# RAG cannot prevent hallucination when the retrieved chunks are marginally related to the query.
# Even if retrieval is "correct" (returns the most similar chunks), those chunks may be about a
# different aspect of the topic. For example, if a user asks "Is coffee bad for your health?" and
# the retriever pulls chunks about coffee sourcing, the model will generate a response using unrelated
# context, sounding confident while answering the wrong question. The core issue: relevance is not
# the same as correctness. A good retrieval score does not guarantee the information answers the
# question being asked.
print("Comment added")
