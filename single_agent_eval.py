import argparse
import csv
import json
import time
from pathlib import Path

import chromadb
import ollama


AGENT_MODEL = "qwen3.5:9b"
EMBED_MODEL = "mxbai-embed-large"


def get_embedding(model_client, text):
    response = model_client.embeddings(
        prompt=text,
        model=EMBED_MODEL
    )
    return response["embedding"]


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


def generate_note_setup(model_client, note_text):
    """
    Generates setup metadata for retrieval/indexing:
    tags + summary.

    Important:
    These are stored in ChromaDB so candidate notes can be represented
    by summaries/tags instead of truncated raw previews.
    """

    system_prompt = """
You are a note metadata generator.

Given a Markdown note, generate:
1. exactly 3 relevant tags
2. a faithful 2-3 sentence summary

Rules:
- Tags must be lowercase and hyphenated.
- The summary must only use information from the note.
- Return ONLY valid JSON.

JSON format:
{
  "tags": ["tag1", "tag2", "tag3"],
  "summary": "2-3 sentence summary."
}
"""

    user_prompt = f"""
Markdown note:
{note_text}
"""

    return call_json_agent(model_client, system_prompt, user_prompt)


def index_vault_with_summary_embeddings(vault_path, model_client, collection):
    """
    Indexes each Markdown note using the note summary embedding.

    ChromaDB document = note summary
    ChromaDB metadata = filename, tags, word_count, character_count

    This matches the current design decision:
    summary embeddings are used to retrieve candidates efficiently,
    especially for larger notes.
    """

    existing_ids = set(collection.get()["ids"])
    indexed = 0
    skipped = 0

    for filepath in Path(vault_path).rglob("*.md"):
        note_id = str(filepath.relative_to(vault_path))

        if note_id in existing_ids:
            skipped += 1
            continue

        note_text = filepath.read_text(errors="ignore")
        setup = generate_note_setup(model_client, note_text)

        tags = setup.get("tags", [])
        summary = setup.get("summary", "")

        if not summary:
            summary = note_text[:1000]

        embedding = get_embedding(model_client, summary)

        collection.add(
            ids=[note_id],
            embeddings=[embedding],
            documents=[summary],
            metadatas=[{
                "filename": note_id,
                "tags": ", ".join(tags),
                "word_count": len(note_text.split()),
                "character_count": len(note_text)
            }]
        )

        indexed += 1
        print(f"Indexed: {note_id}")

    print(f"Indexing complete. Indexed: {indexed}, skipped: {skipped}")


def get_note_setup_from_chromadb(collection, note_id):
    """
    Gets the stored summary and tags for a note from ChromaDB.
    """

    result = collection.get(
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

    tags_string = metadata.get("tags", "")
    tags = [tag.strip() for tag in tags_string.split(",") if tag.strip()]

    return {
        "summary": summary,
        "tags": tags
    }


def query_related_notes(note_id, input_summary, model_client, collection, final_k=3):
    """
    Retrieves top 4 candidates using the input note summary embedding,
    excludes the input note itself, then keeps top 3.

    This avoids returning the note as its own related note.
    """

    total_notes = collection.count()

    if total_notes <= 1:
        return []

    query_embedding = get_embedding(model_client, input_summary)

    # Get top 4 so we can remove the input note and still keep 3 candidates.
    raw_k = min(final_k + 1, total_notes)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=raw_k
    )

    related_notes = []

    for i, candidate_id in enumerate(results["ids"][0]):
        if candidate_id == note_id:
            continue

        candidate_summary = results["documents"][0][i]
        candidate_metadata = results["metadatas"][0][i]

        tags_string = candidate_metadata.get("tags", "")
        candidate_tags = [tag.strip() for tag in tags_string.split(",") if tag.strip()]

        related_notes.append({
            "note_title": candidate_id,
            "summary": candidate_summary,
            "tags": candidate_tags,
            "distance": results["distances"][0][i]
        })

        if len(related_notes) == final_k:
            break

    return related_notes


def run_single_agent(model_client, raw_input_note, candidate_notes):
    """
    Single-agent baseline.

    One general-purpose agent receives the raw input note and candidate
    summaries/tags, then generates tags, summary, and links in one call.
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
- You will receive candidate note summaries and candidate note tags.
- Candidate notes were retrieved using summary embeddings.
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


def run_evaluation(vault_path, output_csv):
    model_client = ollama.Client(host="http://localhost:11434")

    chroma_client = chromadb.PersistentClient(path="./chroma_single_agent_eval")
    collection = chroma_client.get_or_create_collection("single-agent-baseline")

    print("Indexing vault using summary embeddings...")
    index_vault_with_summary_embeddings(vault_path, model_client, collection)

    rows = []

    for filepath in Path(vault_path).rglob("*.md"):
        note_id = str(filepath.relative_to(vault_path))
        raw_input_note = filepath.read_text(errors="ignore")

        print(f"\nProcessing note: {note_id}")

        start_total = time.time()

        try:
            input_setup = get_note_setup_from_chromadb(collection, note_id)
            input_summary = input_setup["summary"]

            candidate_notes = query_related_notes(
                note_id=note_id,
                input_summary=input_summary,
                model_client=model_client,
                collection=collection,
                final_k=3
            )

            start_agent = time.time()
            result = run_single_agent(
                model_client=model_client,
                raw_input_note=raw_input_note,
                candidate_notes=candidate_notes
            )
            agent_latency = time.time() - start_agent

            error = ""

        except Exception as e:
            input_setup = {"summary": "", "tags": []}
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
            "input_setup_summary": input_setup.get("summary", ""),
            "input_setup_tags": json.dumps(input_setup.get("tags", [])),
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
