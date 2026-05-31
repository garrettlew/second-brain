import argparse
import json

import ollama

from main import Agent
from Vault import Vault
from evaluation_helper import run_evaluation


AGENT_MODEL = "qwen3.5:9b"


def safe_json_loads(raw_output):
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1

        if start != -1 and end != -1:
            return json.loads(raw_output[start:end])

        raise


def call_json_agent(model_client, system_prompt, user_prompt):
    response = model_client.chat(
        model=AGENT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        format="json",
        stream=False,
        think=False
    )

    raw_output = response["message"]["content"]
    return safe_json_loads(raw_output)


def run_single_agent(model_client, raw_input_note, candidate_notes):
    """
    Single-agent baseline.

    One general-purpose agent receives:
    - raw input note
    - input note summary
    - input note tags
    - candidate note summaries
    - candidate note tags

    Then it generates tags, summary, and links in one call.
    """

    system_prompt = """
You are a general-purpose Obsidian note enrichment agent.

Your job is to do all note-enrichment tasks in ONE response:
1. Generate exactly 3 relevant tags for the input note.
2. Write a faithful 2-3 sentence summary of the input note.
3. Using the generated tags and summary of the input note, choose up to 3 related notes from the candidate notes.
4. For each selected related note, explain why it is related.

Context design:
- You will receive the raw input note.
- You will receive candidate note summaries and candidate note tags.
- Candidate notes were retrieved using the shared Vault.py ChromaDB setup.
- You must only choose links from the candidate notes.

Rules:
- Tags must be lowercase and hyphenated.
- Do not invent related note filenames.
- If none of the candidate notes are meaningfully related, return an empty list for links.
- Return ONLY valid JSON.

JSON format:
{
  "tags": ["tag1", "tag2", "tag3"],
  "summary": "2-3 sentence summary.",
  "links": [
    {
      "note_title": "filename.md",
      "justification": "One sentence explanation."
    }
  ]
}
"""

    candidate_text = "\n\n".join(
        [
            (
                f"Candidate note: {note['note_title']}\n"
                f"Candidate tags: {', '.join(note['tags'])}\n"
                f"Candidate summary: {note['summary']}"
            )
            for note in candidate_notes
        ]
    )

    user_prompt = f"""
Raw input note:
{raw_input_note}

Candidate related notes:
{candidate_text}
"""

    return call_json_agent(model_client, system_prompt, user_prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vaultpath", type=str, required=True)
    parser.add_argument("--output", type=str, default="single_agent_results.csv")
    args = parser.parse_args()
    model_client = ollama.Client(host="http://localhost:11434")
    agent = Agent(model_client)
    vault_path = args.vaultpath
    vault = Vault(vault_path, model_client, agent)
    run_evaluation(agent, model_client, vault, args.output, run_single_agent)