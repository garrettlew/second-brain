import csv
import json
import time

from pathlib import Path
from Vault import Vault

def run_evaluation(agent, model_client, vault: Vault, output_csv, run_agent_system):
    # model_client = ollama.Client(host="http://localhost:11434")
    #
    # # Shared setup:
    # # Agent is used by Vault.py to generate tags and summaries during indexing.
    # # Vault.py owns the ChromaDB collection and embedding/indexing setup.
    # agent = Agent(model_client)
    # vault = Vault(vault_path, model_client, agent)

    rows = []

    for filepath in Path(vault.vault_path).rglob("*.md"):
        note_id = str(filepath.relative_to(vault.vault_path))
        raw_input_note = filepath.read_text(errors="ignore")

        print(f"\nProcessing note: {note_id}")

        start_total = time.time()

        try:
            input_setup = get_note_setup_from_vault(vault, note_id)
            input_summary = input_setup.get("summary", "")
            input_tags = input_setup.get("tags", [])
            input_embedding = input_setup.get("embedding", [])

            candidate_notes = query_related_notes_from_vault(
                vault=vault,
                note_id=note_id,
                query_embedding=input_embedding,
                final_k=3
            )

            start_agent = time.time()

            result = run_agent_system(
                model_client=model_client,
                raw_input_note=raw_input_note,
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


def get_note_setup_from_vault(vault, note_id):
    """
    Get the stored summary and tags for the input note from the shared Vault collection.
    In Vault.py, the ChromaDB document is the generated summary, and tags are stored in metadata.
    """

    result = vault.collection.get(
        ids=[note_id],
        include=["embeddings", "documents", "metadatas"]
    )

    if not result["ids"]:
        return {
            "summary": "",
            "tags": []
        }

    summary = result["documents"][0]
    metadata = result["metadatas"][0]
    embedding = result['embeddings'][0]

    return {
        "embedding": embedding,
        "summary": summary,
        "tags": parse_tags(metadata)
    }

def query_related_notes_from_vault(vault, note_id, query_embedding, final_k=3):
    """
    Query the shared Vault ChromaDB collection.

    We retrieve top 4 candidates, remove the input note itself if it appears,
    and keep the top 3 real candidate notes.
    """

    total_notes = vault.collection.count()

    if total_notes <= 1:
        return []

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


def parse_tags(metadata):
    tags_string = metadata.get("tags", "")
    return [tag.strip() for tag in tags_string.split(",") if tag.strip()]