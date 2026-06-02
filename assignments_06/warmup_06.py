from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from dotenv import load_dotenv
import os
import string
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

os.system('cls')

# =====================================================================================
#                    Week 6 - Part 1: Warm up Exercise
# =====================================================================================

print('='*100)
print("--------------- Part 1: Warmup Exercises ---------------")
print('='*100)

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# =====================================================================================
# RAG Concepts
# =====================================================================================
print('='*100)
print("--------------- RAG Concepts ---------------")
# print('='*100)

# =====================================================================================
# Concepts Question 1
# =====================================================================================

print('-'*100)
print("--------------- Concepts Question 1 ---------------")
print('-'*100)


print("\nComment added")

# Scenario A:
#
# Legal team with hundreds of PDFs that get updated every quarter.
# I'd go with RAG here. The whole point is that these documents change regularly,
# so you don't want to bake the answers into the model itself. With RAG, the system
# just pulls the right sections from whatever the latest version of the docs is.
# No retraining needed when policies change.
#
# Scenario B:
#
# Startup that needs copy written in a very specific brand voice, with 3,000 examples.
# Fine-tuning makes the most sense. The brand voice is unusual enough that you can't
# just describe it in a prompt and expect the model to nail it. But with 3,000 real
# examples to learn from, fine-tuning can teach the model what that voice actually
# sounds like so it can reproduce it naturally.
#
# Scenario C:
#
# Analyst asking questions about one short two-page report.
# Prompt engineering — just paste the report into the prompt. It's short enough to
# fit easily, and she only needs this once. Building a whole retrieval pipeline for
# two pages would be overkill.

# =====================================================================================
# Concepts Question 2
# =====================================================================================

print('-'*100)
print("--------------- Concepts Question 2 ---------------")
print('-'*100)
print("\nComment added")
# Why is a confidently wrong answer worse than one that admits uncertainty?
#
# When a model says "I'm not sure," you know to go check for yourself. But when it
# states something like it's a fact, you tend to just trust it and move on. That's
# what makes hallucinations tricky — the model doesn't hedge or qualify, so there's
# no red flag telling you to double-check.
#
# A real-world example: imagine a medical chatbot confidently tells someone it's fine
# to take two medications together, when actually they interact badly. If the model
# had said "I'm not certain, check with your pharmacist," the person would have asked.
# But the confident tone made it sound like settled medical advice, so they didn't.

# =====================================================================================
# Concepts Question 3
# =====================================================================================

print('-'*100)
print("--------------- Concepts Question 3 ---------------")
print('-'*100)

print("RAG pipeline steps in the Out-of-order:")
print("""steps = [
    "Generate a response from the LLM",
    "Extract text from source documents",
    "Receive the user's query",
    "Retrieve the most relevant chunks",
    "Convert text chunks into embeddings",
    "Inject retrieved chunks into the prompt",
    "Split text into chunks",
    "Embed the user's query"]""")

print('-'*100)
print("RAG pipeline steps in the correct order:")
print("""steps = [
    "Extract text from source documents",
    "Split text into chunks",
    "Convert text chunks into embeddings",
    "Receive the user's query",
    "Embed the user's query",
    "Retrieve the most relevant chunks",
    "Inject retrieved chunks into the prompt",    
    "Generate a response from the LLM"]""")
# RAG pipeline steps in the correct order:
#
# 1. Extract text from source documents
#     — Read the raw content out of PDFs, web pages, or whatever format the documents are in.
#
# 2. Split text into chunks
#     — Break the full text into smaller pieces so we can retrieve just the relevant parts later.
#
# 3. Convert text chunks into embeddings
#     — Turn each chunk into a numerical vector that captures its meaning, then store those in an index.
#
# 4. Receive the user's query
#     — The user types a question they want answered.
#
# 5. Embed the user's query
#     — Convert the question into the same kind of vector so we can compare it against the stored chunks.
#
# 6. Retrieve the most relevant chunks
#     — Find the chunks whose embeddings are closest to the query embedding (e.g. using cosine similarity).
#
# 7. Inject retrieved chunks into the prompt
#     — Paste those chunks into the prompt as context so the model has real information to work with.
#
# 8. Generate a response from the LLM
#     — The model reads the prompt (question + retrieved context) and produces a grounded answer.


# =====================================================================================
# Keyword RAG
# =====================================================================================
print('='*100)
print("--------------- Keyword RAG ---------------")
# print('='*100)


def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content)
                for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

# =====================================================================================
# Keyword Question 1
# =====================================================================================


print('-'*100)
print("--------------- Keyword Question 1 ---------------")
print('-'*100)

query = "What are your hours on the weekend?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

result = simple_keyword_retrieval(query, documents, verbose=True)
# print(f"Sel: {result}")
print(f"Qu 1: Selected Document: {result[0][0]}")

# Selected Document is "hours.txt" as it has the strongest token overlap with query
# query token 'hours' and 'weekend' both appears in hours.txt

# =====================================================================================
# Keyword Question 2
# =====================================================================================

print('-'*100)
print("--------------- Keyword Question 2 ---------------")
print('-'*100)

# Run the same function with this second query using the same documents from Q1:
query = "Do you have anything without caffeine?"

result = simple_keyword_retrieval(query, documents, verbose=True)
print(f"Qu 2: Selected Document: {result[0][0]}")

# No document was selected here, so keyword RAG did not get this right.
# The query is asking for caffeine-free choices, but none of the documents use those exact words,
# so the overlap scorer has nothing to latch onto.
# A semantic retriever would do better because it can connect "without caffeine" to ideas like
# decaf or caffeine-free drinks even when the wording is different.

# =====================================================================================
# Keyword Question 3
# =====================================================================================

print('-'*100)
print("--------------- Keyword Question 3 ---------------")
print('-'*100)

query = "How do I sign up for rewards?"

# Prediction:
# This will probably retun "None found" as the query talks about "rewards"

result = simple_keyword_retrieval(query, documents, verbose=True)
print(f"Qu 3: Selected Document: {result[0][0]}")

# The result was "None found," which matches the exact keyword overlap logic here.
# The query says "rewards," but the documents talk about loyalty programs, points, and drinks,
# so there is no shared token for the retriever to match.

# =====================================================================================
# Semantic RAG Concepts
# =====================================================================================
print('='*100)
print("--------------- Semantic RAG Concepts ---------------")
# print('='*100)

# =====================================================================================
# Semantic Question 1
# =====================================================================================

print('-'*100)
print("--------------- Semantic Question 1 ---------------")
print('-'*100)

# What is a vector embedding?
# A vector embedding is just a list of numbers that the model uses to represent the meaning
# of a piece of text. The idea is that texts with similar meanings end up with similar sets
# of numbers, so you can compare them mathematically even if they use totally different words.

# Cosine similarity scores of 0.85 vs 0.30 — which chunk is more relevant?
# The chunk with 0.85 is far more relevant. Cosine similarity measures how closely two vectors
# point in the same direction, so a score close to 1 means the two texts are talking about
# nearly the same thing. A score of 0.30 means they share some vague relationship but are
# mostly pointing in different directions — not a useful match.

# Why can semantic search find relevant chunks even when no exact words match?
# Because it is comparing the meaning encoded in the embeddings, not the literal characters.
# If the query says "cost" and the document says "price," they still end up with similar
# embeddings because the model learned that those words are used in similar contexts.
# Keyword search would miss that entirely.

print("Comment added")

# =====================================================================================
# Semantic Question 2
# =====================================================================================

print('-'*100)
print("--------------- Semantic Question 2 ---------------")
print('-'*100)

# Comparison table: keyword RAG vs semantic RAG
#
# | Feature                    | Keyword RAG                       | Semantic RAG                              |
# |----------------------------|-----------------------------------|-------------------------------------------|
# | What is compared?          | Exact word overlap                | Vector embeddings (meaning of the text)   |
# | What is retrieved?         | Full document                     | The most relevant chunks from an index    |
# | Can it handle synonyms?    | No                                | Yes — similar meanings score as similar   |
# | Storage format             | Plain text dictionary             | Vector store / embedding index            |
# | Relevance score            | Number of overlapping keywords    | Cosine similarity between query and chunk |

print("Comment added")

# =====================================================================================
# LlamaIndex
# =====================================================================================
print('='*100)
print("--------------- LlamaIndex ---------------")
# print('='*100)

# =====================================================================================
# LlamaIndex Question 1
# =====================================================================================

print('-'*100)
print("--------------- LlamaIndex Question 1 ---------------")
print('-'*100)

brightleaf_path = "../../python-200/lessons/06_AI_augmentation/resources/brightleaf_pdfs"

documents = SimpleDirectoryReader(brightleaf_path).load_data()
index = VectorStoreIndex.from_documents(documents)

# Query Engine
query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for qu in questions:
    print(f"\n{'-'*100}")
    print(f"Question: {qu}")

    response = query_engine.query(qu)
    print(f"Answer: {response}")

    print(f"\nSource nodes:")
    for i, node in enumerate(response.source_nodes, 1):
        score = node.score if node.score is not None else "N/A"
        snippet = node.get_content()[:150].replace("\n", " ")
        print(f" [{i}] score = {score:.4f} text = {snippet}")

# Query 1: "What employee benefits does BrightLeaf offer?"
# Retrieved chunks: The top 3 nodes should come from the HR or employee handbook PDF,
# covering health insurance, retirement plans, and similar benefit descriptions.
# Those look directly relevant to the question.
# Model tone: LlamaIndex responses tend to be specific and grounded when the source material
# is clear, so I expect the answer to cite concrete benefits rather than hedging.
# Unexpected: Would be surprised if a security-policy or installation-guide chunk showed up here,
# but keyword proximity could pull in something adjacent like "employee responsibilities."

# Query 2: "What are BrightLeaf's security policies?"
# Retrieved chunks: Should pull from whatever document covers IT/physical security rules.
# If BrightLeaf's PDFs don't have a dedicated security policy doc, the retriever might grab
# tangentially related chunks (e.g. data privacy sections from the employee handbook).
# Model tone: Expect confident, specific language if the source material is detailed;
# more hedging ("based on the provided context") if the relevant content is sparse.
# Unexpected: Installation or maintenance PDFs might appear if the word "security" shows up
# in a physical-safety context rather than an IT-security context.

# =====================================================================================
# LlamaIndex Question 2
# =====================================================================================

print('-'*100)
print("--------------- LlamaIndex Question 2 ---------------")
print('-'*100)

# Re-run one of the queries from Q1 twice: once with similarity_top_k=1 and once with similarity_top_k=5

comapre_qu = "What employee benefits does BrightLeaf offer?"

for top_k in [1, 5]:
    print(f"\n{'-'*100}")
    print(f"Q2 test -- similarity_top_k = {top_k}")
    print(f"Question: {comapre_qu}")

    engine = index.as_query_engine(similarity_top_k=top_k)
    response = engine.query(comapre_qu)
    print(f"Answer: {response}")

    print(f"\nSource nodes:")
    for i, node in enumerate(response.source_nodes, 1):
        score = node.score if node.score is not None else "N/A"
        snippet = node.get_content()[:150].replace("\n", " ")
        print(f" [{i}] score = {score:.4f} text = {snippet}")

# Observation:
#
# With top_k=1, the answer is usually tighter and sticks closely to one high-confidence chunk.
# With top_k=5, the answer can become more complete, but it can also pick up extra details that
# are less central to the question. More retrieved context is not always better: if additional
# chunks are noisy or only loosely related, they can dilute or distract from the best answer.

# =====================================================================================
# LlamaIndex Question 3
# =====================================================================================

print('-'*100)
print("--------------- LlamaIndex Question 3 ---------------")
print('-'*100)

# Try a query that the pipeline may struggle with.
hard_query = "How does BrightLeaf compare to other solar companies in market share and customer satisfaction?"

print(f"\n{'-'*60}")
print("Q3 hard query")
print(f"Question: {hard_query}")

# Use a higher top_k here to inspect more retrieved evidence.
hard_engine = index.as_query_engine(similarity_top_k=5)
hard_response = hard_engine.query(hard_query)

print(f"\nAnswer:\n{hard_response}")
print("\nAll retrieved chunks:")
for i, node in enumerate(hard_response.source_nodes, 1):
    score_text = f"{node.score:.4f}" if node.score is not None else "N/A"
    snippet = node.get_content()[:150].replace("\n", " ")
    print(f"  [{i}] score={score_text}  text: {snippet}...")

# Reflection:
# Expected:
# This query should be hard because it asks for market-share/customer-satisfaction
# comparisons that may not exist in internal BrightLeaf policy/handbook PDFs.
# Observed behavior: The retriever will likely return loosely related company or policy chunks,
# and the model may hedge or answer partially from limited context.
# Improvement ideas: Add external industry data sources, use metadata filtering by document type,
# and add a confidence/abstention rule so the system says "not enough evidence" when context is weak.

# =====================================================================================
# LlamaIndex Question 4
# =====================================================================================

print('-'*100)
print("--------------- LlamaIndex Question 4 ---------------")
print('-'*100)

# Evaluate one strong query and one likely weak query using built-in evaluators.

judge_llm = OpenAI(model="gpt-4o-mini")
faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

# Query A: expected stronger response quality.
q = "What employee benefits does BrightLeaf offer?"
resp_a = query_engine.query(q)

faith_a = faithfulness_evaluator.evaluate_response(query=q, response=resp_a)
rel_a = relevancy_evaluator.evaluate_response(query=q, response=resp_a)

faith_a_score = 1.0 if getattr(faith_a, "passing", False) else 0.0
rel_a_score = 1.0 if getattr(rel_a, "passing", False) else 0.0

print(f"\n{'-'*60}")
print("Q4 evaluation - Query A")
print(f"Query: {q}")
print(f"Faithfulness score: {faith_a_score}")
print(f"Relevancy score: {rel_a_score}")

# Query B: expected lower quality because information likely is not in the docs.
q_bad = "What is BrightLeaf's stock ticker and current quarterly revenue guidance?"
resp_b = query_engine.query(q_bad)

faith_b = faithfulness_evaluator.evaluate_response(
    query=q_bad, response=resp_b)
rel_b = relevancy_evaluator.evaluate_response(query=q_bad, response=resp_b)

faith_b_score = 1.0 if getattr(faith_b, "passing", False) else 0.0
rel_b_score = 1.0 if getattr(rel_b, "passing", False) else 0.0

print(f"\n{'-'*60}")
print("Q4 evaluation - Query B")
print(f"Query: {q_bad}")
print(f"Faithfulness score: {faith_b_score}")
print(f"Relevancy score: {rel_b_score}")

# Q4 analysis:

# 1) Faithfulness 1.0 means the answer is supported by retrieved context and does not appear to
# invent facts beyond that evidence. A faithfulness score of 0.0 indicates unsupported claims or
# contradictions with the provided source chunks.

# 2) Relevancy measures whether the answer actually addresses the user question.
# It differs from faithfulness because an answer can be faithful to context but still off-topic.

# 3) Scores should usually be higher for the benefits query and lower for the stock/revenue query,
# because the second question likely has little or no supporting evidence in these PDFs.

# 4) LLM-as-a-judge means using another model to grade quality dimensions (like faithfulness and
# relevance) from natural language outputs. It is used in RAG because there is often no single
# exact "correct answer" string, so simple accuracy metrics are too brittle.
