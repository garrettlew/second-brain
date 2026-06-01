import csv
import json
import platform
import psutil
import resource
import threading
import time

from pathlib import Path
from Vault import Vault


def run_evaluation(vault: Vault, output_csv, run_agent_system):
    """
    Run the experiment on the given run_agent_system function

    Track latency and peak memory usage. And save the outputs to a csv file.
    """

    rows = []

    for filepath in Path(vault.vault_path).rglob("*.md"):
        note_id = str(filepath.relative_to(vault.vault_path))
        raw_input_note = filepath.read_text(errors="ignore")

        print(f"\nProcessing note: {note_id}")

        start_total = time.time()
        peak_rss_client = None
        peak_rss_ollama = None

        try:
            input_setup = vault.get_note_setup_from_vault(note_id)
            input_summary = input_setup.get("summary", "")
            input_tags = input_setup.get("tags", [])
            input_embedding = input_setup.get("embedding", [])

            candidate_notes = vault.query_related_notes_from_vault(
                note_id=note_id,
                query_embedding=input_embedding,
                final_k=3
            )

            candidate_data = [remove_key(d, 'distance') for d in candidate_notes]

            start_agent = time.time()
            rss_before = peak_rss_mb()
            ollama_result, stop_event, sampler_thread = sample_ollama_peak_mb()

            try:
                result = run_agent_system(
                    raw_input_note=raw_input_note,
                    candidate_notes=candidate_data
                )
            finally:
                stop_event.set()
                sampler_thread.join(timeout=1.0)

            agent_latency = time.time() - start_agent
            peak_rss_client = round(peak_rss_mb() - rss_before, 2)
            peak_rss_ollama = round(ollama_result[0], 2) if ollama_result[0] is not None else None
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


def remove_key(d, key):
    new_d = d.copy()      # Create a shallow copy (fast C-level operation)
    new_d.pop(key, None)  # Remove the key safely (O(1) operation)
    return new_d
