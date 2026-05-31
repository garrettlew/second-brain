import argparse
import json
<<<<<<< HEAD
import platform
import resource
import threading
import time
from pathlib import Path
=======
>>>>>>> 892eb2aecb095f5494bcd7724995c46fc6303e41

import ollama
import psutil

from main import Agent
from Vault import Vault
from evaluation_helper import run_evaluation


AGENT_MODEL = "qwen3.5:9b"


<<<<<<< HEAD
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
=======
def safe_json_loads(raw_output):
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1

        if start != -1 and end != -1:
            return json.loads(raw_output[start:end])

        raise
>>>>>>> 892eb2aecb095f5494bcd7724995c46fc6303e41


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

<<<<<<< HEAD
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
=======
    return call_json_agent(model_client, system_prompt, user_prompt)
>>>>>>> 892eb2aecb095f5494bcd7724995c46fc6303e41


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
