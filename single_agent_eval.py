import argparse
import csv
import json
import platform
import resource
import threading
import time
from pathlib import Path

import chromadb
import ollama
import psutil


AGENT_MODEL = "qwen3.5:9b"
EMBED_MODEL = "mxbai-embed-large"


def peak_rss_mb() -> float:
    """Return peak resident set size of this process in MB (macOS returns bytes, Linux returns KiB)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return rss / 1024 / 1024
    return rss / 1024


def sample_ollama_peak_mb() -> tuple[float | None, threading.Event]:
    """
    Start a background thread that polls the ollama process RSS every 50 ms.
    Returns (result_container, stop_event). After stopping, result_container[0]
    holds the peak RSS in MB, or None if ollama was not found.
    """
    stop_event = threading.Event()
    result = [None]

    def _sample():
        peak = 0.0
        ollama_proc = None
        for proc in psutil.process_iter(["name", "pid"]):
            if "ollama" in proc.info["name"].lower():
                try:
                    ollama_proc = psutil.Process(proc.info["pid"])
                except psutil.NoSuchProcess:
                    pass
                break
        if ollama_proc is None:
            return
        while not stop_event.is_set():
            try:
                rss = ollama_proc.memory_info().rss / 1024 / 1024
                if rss > peak:
                    peak = rss
            except psutil.NoSuchProcess:
                break
            stop_event.wait(0.05)
        result[0] = peak if peak > 0 else None

    t = threading.Thread(target=_sample, daemon=True)
    t.start()
    return result, stop_event, t


def get_embedding(model_client, text):
    response = model_client.embeddings(
        prompt=text,
        model=EMBED_MODEL
    )
    return response["embedding"]


def index_vault(vault_path, model_client, collection):
    existing_ids = set(collection.get()["ids"])
    indexed = 0
    skipped = 0

    for filepath in Path(vault_path).rglob("*.md"):
        note_id = str(filepath.relative_to(vault_path))

        if note_id in existing_ids:
            skipped += 1
            continue

        note_text = filepath.read_text(errors="ignore")
        embedding = get_embedding(model_client, note_text)

        collection.add(
            ids=[note_id],
            embeddings=[embedding],
            documents=[note_text],
            metadatas=[{"filename": note_id}]
        )

        indexed += 1
        print(f"Indexed: {note_id}")

    print(f"Indexing complete. Indexed: {indexed}, skipped: {skipped}")


def query_related_notes(note_id, note_text, model_client, collection, top_k=3):
    total_notes = collection.count()

    if total_notes <= 1:
        return []

    embedding = get_embedding(model_client, note_text)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k + 1, total_notes)
    )

    related_notes = []

    for i, candidate_id in enumerate(results["ids"][0]):
        if candidate_id == note_id:
            continue

        related_notes.append({
            "note_title": candidate_id,
            "content_preview": results["documents"][0][i][:700],
            "distance": results["distances"][0][i]
        })

        if len(related_notes) == top_k:
            break

    return related_notes


def run_single_agent(model_client, note_text, candidate_notes):
    system_prompt = """
You are a general-purpose Obsidian note enrichment agent.

Your job is to do all note-enrichment tasks in ONE response:
1. Generate exactly 3 relevant tags.
2. Write a faithful 2-3 sentence summary.
3. Choose up to 3 related notes from the provided candidate notes.
4. For each related note, explain why it is related.

Rules:
- Tags must be lowercase and hyphenated.
- Do not invent related notes.
- Only select related notes from the candidate notes.
- If none of the candidate notes are meaningfully related, return an empty list for links.
- Return ONLY valid JSON.

Return JSON in this exact format:
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
            f"Candidate note: {note['note_title']}\nPreview: {note['content_preview']}"
            for note in candidate_notes
        ]
    )

    user_prompt = f"""
Input note:
{note_text}

Candidate related notes:
{candidate_text}
"""

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
    return json.loads(raw_output)


def run_evaluation(vault_path, output_csv):
    model_client = ollama.Client(host="http://localhost:11434")

    chroma_client = chromadb.PersistentClient(path="./chroma_single_agent_eval")
    collection = chroma_client.get_or_create_collection("single-agent-baseline")

    print("Indexing vault into ChromaDB...")
    index_vault(vault_path, model_client, collection)

    rows = []

    for filepath in Path(vault_path).rglob("*.md"):
        note_id = str(filepath.relative_to(vault_path))
        note_text = filepath.read_text(errors="ignore")

        print(f"\nProcessing note: {note_id}")

        candidate_notes = query_related_notes(
            note_id=note_id,
            note_text=note_text,
            model_client=model_client,
            collection=collection,
            top_k=3
        )

        start_time = time.time()
        rss_before = peak_rss_mb()
        ollama_result, stop_event, sampler_thread = sample_ollama_peak_mb()

        try:
            result = run_single_agent(model_client, note_text, candidate_notes)
            error = ""
        except Exception as e:
            result = {
                "tags": [],
                "summary": "",
                "links": []
            }
            error = str(e)

        stop_event.set()
        sampler_thread.join(timeout=1.0)
        latency = time.time() - start_time
        peak_rss_client = round(peak_rss_mb() - rss_before, 2)
        peak_rss_ollama = round(ollama_result[0], 2) if ollama_result[0] is not None else None

        rows.append({
            "condition": "single_agent",
            "note_id": note_id,
            "candidate_notes": json.dumps(candidate_notes),
            "tags": json.dumps(result.get("tags", [])),
            "summary": result.get("summary", ""),
            "links": json.dumps(result.get("links", [])),
            "latency_seconds": round(latency, 3),
            "peak_rss_client_mb": peak_rss_client,
            "peak_rss_ollama_mb": peak_rss_ollama,
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
