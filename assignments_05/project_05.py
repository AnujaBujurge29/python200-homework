from dotenv import load_dotenv
from openai import OpenAI
import json

import os
os.system('cls')

# ------------------------------------------------------------------------------------------------------
# Week 5 - Part 2: Mini-Project — Job Application Helper
# ------------------------------------------------------------------------------------------------------

print('*'*150)
print("                                              Week 5 - Part 2: Mini-Project — Job Application Helper")
print('*'*150)

# ------------------------------------------------------------------------------------------------------
# Task 1: Setup and System Prompt
# ------------------------------------------------------------------------------------------------------

print("Task 1: Setup and System Prompt")
print('-'*100)

load_dotenv()
client = OpenAI()


def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content


system_prompt = """You are a professional job application coach specializing in helping career changers \
rewrite their resume bullet points and craft compelling cover letters for new industries.

Your audience is professionals transitioning from one field to another — for example, a teacher \
moving into data engineering, or a nurse moving into project management. They have real skills \
but need help translating their experience into language that resonates with hiring managers \
in their target field.

Guidelines:
- Stay focused exclusively on job application materials: resumes, cover letters, bullet points, \
and interview prep. Politely decline requests outside this scope.
- When rewriting bullet points, use strong action verbs and quantify impact whenever possible.
- Always remind the user to review and edit your output before submitting it anywhere — you are \
a drafting tool, not a final editor.
- Acknowledge that you may not know the user's specific industry norms or company culture. \
Encourage the user to use their own judgment and research to adapt your suggestions.
- Ask clarifying questions when the user's request is vague — for example, ask about their \
target role, industry, or what skills they want to highlight.
- Keep a supportive, professional tone without being overly flattering or generic."""

messages = [{"role": "system", "content": system_prompt}]

print("System prompt loaded successfully.")
print(f"\nSystem Prompt:\n{system_prompt}")

# Deliberate choice: I specified the audience as "career changers" transitioning between fields
# rather than general job seekers. This gives the model a concrete persona to help, which leads
# to more targeted advice — e.g., it will focus on translating transferable skills rather than
# just polishing existing industry-specific language.

# ------------------------------------------------------------------------------------------------------
# Task 2: Bullet Point Rewriter
# ------------------------------------------------------------------------------------------------------

print('-'*100)
print("Task 2: Bullet Point Rewriter")
print('-'*100)


def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion(), parse the JSON, and return the result
    raw_response = get_completion(messages, temperature=0.7)

    # Strip markdown code fences if present
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]  # remove first line (```json)
        cleaned = cleaned.rsplit("```", 1)[0]  # remove closing ```
        cleaned = cleaned.strip()

    try:
        results = json.loads(cleaned)
        for item in results:
            print(f"Original: {item['original']}")
            print(f"Improved: {item['improved']}")
            print()
        return results
    except json.JSONDecodeError:
        print(f"Failed to Parsed JSON Raw response.")
        print(raw_response)
        return []


bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

rewrittten = rewrite_bullets(bullets)

# These bullets are weak because they use vague verbs ("helped", "made", "worked"), lack
# quantifiable results, and don't specify what was actually accomplished. The model typically
# suggests stronger action verbs (e.g., "resolved", "developed", "collaborated"), adds
# specificity about the type of work, and frames outcomes in terms of impact or results.

# ------------------------------------------------------------------------------------------------------
# Task 3: Cover Letter Generator
# ------------------------------------------------------------------------------------------------------

print('-'*100)
print("Task 3: Cover Letter Generator")
print('-'*100)


def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion() and return the result
    return get_completion(messages, temperature=0.7)


job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."

cover_letter = generate_cover_letter(job_title, background)
print("Cover letter generated:\n")
print(cover_letter)

# I chose examples featuring career changers from very different fields (nursing → data, banking → software)
# to show the model that it should bridge the gap between a previous career and a new one.
# Both examples demonstrate a confident tone that frames past experience as an asset rather than
# a limitation, and they avoid generic phrases like "I am a hard worker."
# The few-shot pattern helps control tone, length, and structure — without examples, the model
# might produce overly formal or generic openings that don't feel tailored to a career changer.

# ------------------------------------------------------------------------------------------------------
# Task 4: Moderation Check
# ------------------------------------------------------------------------------------------------------

print('-'*100)
print("Task 4: Moderation Check")
print('-'*100)


def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    # Your code here: return True if safe, False if flagged, and print a message if flagged
    if flagged:
        print("Input is flagged, please rephrase request and try again.")
        return False
    return True


# Test with safe input
safe_text = "I need help in rewritting my resume for Data Engineering Role"
print(f"Safe test: '{safe_text}'")
print(f"Result: is_safe = {is_safe(safe_text)}")

# Test with flagged input
flagged_text = "I need help in planning bank robbery"
print(f"Flagged test: '{flagged_text}'")
print(f"Result: is_safe = {is_safe(flagged_text)}")

# ------------------------------------------------------------------------------------------------------
# Task 5: The Chatbot Loop
# ------------------------------------------------------------------------------------------------------

print('-'*100)
print("Task 5: The Chatbot Loop")
print('-'*100)


def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print(
                "\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            # YOUR CODE: call rewrite_bullets() and print the result
            if raw_bullets:
                print("\nJob Application Helper: Here are your imporved bullet points:")
                rewrite_bullets(raw_bullets)
            else:
                print(
                    "\nJob Application Helper: No bullet points recived, Please try again later..")

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input(
                "Job Application Helper: What is the job title? ").strip()
            background = input(
                "Job Application Helper: Briefly describe your background: ").strip()
            # YOUR CODE: call generate_cover_letter() and print the result
            if job_title and background:
                print("\nJob Application Helper: This is draft cover letter:")
                result = generate_cover_letter(job_title, background)
                print(result)
            else:
                print(
                    "\nJob Application Helper: I need both Job Title and background to generate cover letter")

        # 7. Otherwise, handle it as a regular chat turn
        else:
            # YOUR CODE:
            # - Append the user's message to `messages`
            # - Call get_completion(messages)
            # - Print the reply
            # - Append the reply to `messages` as an assistant message
            messages.append({"role": "user", "content": user_input})
            reply = get_completion(messages)
            print(f"\nJob Application Helper: {reply}")
            messages.append({"role": "user", "content": reply})


if __name__ == "__main__":
    run_chatbot()

# ------------------------------------------------------------------------------------------------------
# Task 6: Ethics Reflection (Option A - Comment Block)
# ------------------------------------------------------------------------------------------------------

# Format chosen: Option A - Comment block
#
# 1. Bias in training data:
# The model was trained on text that likely overrepresents certain industries (tech, finance),
# communication styles (Western, corporate), and cultural backgrounds. This means its resume
# and cover letter advice may favor a polished, American-corporate tone and undervalue
# communication styles common in other cultures or non-traditional industries. For example,
# it might suggest removing collaborative language that is valued in some cultures, or push
# a self-promotional tone that feels unnatural to many job-seekers.
#
# 2. Risks of submitting output without review:
# If a job-seeker submitted the bot's output directly without reviewing it, several things could
# go wrong. The model might invent or exaggerate qualifications the user never mentioned, use
# phrasing that doesn't match the user's actual voice, or include details that are inaccurate
# for their specific industry. An employer could also notice the generic, AI-generated tone,
# which could hurt the applicant's credibility. The bot is a drafting tool, not a final product.
#
# 3. Guardrail for professional deployment:
# If deploying this tool professionally, I would add a prominent disclaimer at the top of every
# generated output stating: "This is an AI-generated draft. Review carefully, verify all facts,
# and edit to match your voice before submitting." Additionally, I would log flagged moderation
# events and implement rate limiting to prevent misuse of the API.