import argparse
import csv
import json
import time
from pathlib import Path

import ollama

from main import Agent
from Vault import Vault


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


def parse_tags(metadata):
    tags_string = metadata.get("tags", "")
    return [tag.strip() for tag in tags_string.split(",") if tag.strip()]


def get_note_setup_from_vault(vault, note_id):
    """
    Get the stored summary and tags for the input note from the shared Vault collection.
    In Vault.py, the ChromaDB document is the generated summary, and tags are stored in metadata.
    """

    result = vault.collection.get(
        ids=[note_id],
        include=["documents", "metadatas"]
    )

    if not result["ids"]:
        return {
            "summary": "",
            "tags": []
        }

    summary = result["documents"][0]
    metadata = result["metadatas"][0]

    return {
        "summary": summary,
        "tags": parse_tags(metadata)
    }


def get_embedding_from_vault(vault, text):
    """
    Use the same embedding model/client from Vault.py.
    """

    response = vault.model_client.embeddings(
        prompt=text,
        model=vault.model_type
    )

    return response["embedding"]


def query_related_notes_from_vault(vault, note_id, input_summary, final_k=3):
    """
    Query the shared Vault ChromaDB collection.

    We retrieve top 4 candidates, remove the input note itself if it appears,
    and keep the top 3 real candidate notes.
    """

    total_notes = vault.collection.count()

    if total_notes <= 1:
        return []

    query_embedding = get_embedding_from_vault(vault, input_summary)

    raw_k = min(final_k + 1, total_notes)

    results = vault.collection.query(
        query_embeddings=[query_embedding],
        n_results=raw_k,
        include=["documents", "metadatas", "distances"]
    )

    related_notes = []

    for i, candidate_id in enumerate(results["ids"][0]):
        if candidate_id == note_id:
            continue

        candidate_summary = results["documents"][0][i]
        candidate_metadata = results["metadatas"][0][i]

        related_notes.append({
            "note_title": candidate_id,
            "summary": candidate_summary,
            "tags": parse_tags(candidate_metadata),
            "distance": results["distances"][0][i]
        })

        if len(related_notes) == final_k:
            break

    return related_notes


def run_single_agent(model_client, raw_input_note, input_summary, input_tags, candidate_notes):
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
3. Choose up to 3 related notes from the candidate notes.
4. For each selected related note, explain why it is related.

Context design:
- You will receive the raw input note.
- You will also receive the input note's setup summary and setup tags.
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

Input note setup summary:
{input_summary}

Input note setup tags:
{', '.join(input_tags)}

Candidate related notes:
{candidate_text}
"""

    return call_json_agent(model_client, system_prompt, user_prompt)


def run_evaluation(vault_path, output_csv):
    model_client = ollama.Client(host="http://localhost:11434")

    # Shared setup:
    # Agent is used by Vault.py to generate tags and summaries during indexing.
    # Vault.py owns the ChromaDB collection and embedding/indexing setup.
    agent = Agent(model_client)
    vault = Vault(vault_path, model_client, agent)

    rows = []

    for filepath in Path(vault_path).rglob("*.md"):
        note_id = str(filepath.relative_to(vault_path))
        raw_input_note = filepath.read_text(errors="ignore")

        print(f"\nProcessing note: {note_id}")

        start_total = time.time()

        try:
            input_setup = get_note_setup_from_vault(vault, note_id)
            input_summary = input_setup.get("summary", "")
            input_tags = input_setup.get("tags", [])

            candidate_notes = query_related_notes_from_vault(
                vault=vault,
                note_id=note_id,
                input_summary=input_summary,
                final_k=3
            )

            start_agent = time.time()

            result = run_single_agent(
                model_client=model_client,
                raw_input_note=raw_input_note,
                input_summary=input_summary,
                input_tags=input_tags,
                candidate_notes=candidate_notes
            )

            agent_latency = time.time() - start_agent
            error = ""

        except Exception as e:
            input_summary = ""
            input_tags = []
            candidate_notes = []
            result = {
                "tags": [],
                "summary": "",
                "links": []
            }
            agent_latency = 0
            error = str(e)

        total_latency = time.time() - start_total

        rows.append({
            "condition": "single_agent",
            "note_id": note_id,
            "word_count": len(raw_input_note.split()),
            "character_count": len(raw_input_note),
            "input_setup_summary": input_summary,
            "input_setup_tags": json.dumps(input_tags),
            "candidate_notes": json.dumps(candidate_notes),
            "generated_tags": json.dumps(result.get("tags", [])),
            "generated_summary": result.get("summary", ""),
            "generated_links": json.dumps(result.get("links", [])),
            "agent_latency_seconds": round(agent_latency, 3),
            "total_latency_seconds": round(total_latency, 3),
            "tag_relevance_score": "",
            "summary_faithfulness_score": "",
            "link_quality_score": "",
            "comments": "",
            "error": error
        })

    if not rows:
        print("No Markdown notes found in the vault.")
        return

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved evaluation results to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vaultpath", type=str, required=True)
    parser.add_argument("--output", type=str, default="single_agent_results.csv")
    args = parser.parse_args()

    run_evaluation(args.vaultpath, args.output)
