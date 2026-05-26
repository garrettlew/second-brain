import argparse
import chromadb
import json
import ollama
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_prompt(category: str, name: str) -> str:
    return (Path(__file__).parent / "prompts" / category / f"{name}.txt").read_text()


def log_experiment(category: str, payload: dict):
    out_dir = Path(__file__).parent / "experiments"
    out_dir.mkdir(exist_ok=True)
    with (out_dir / f"{category}.jsonl").open("a") as f:
        f.write(json.dumps(payload) + "\n")


def peak_rss_mb() -> float:
    """Return peak RSS in MB. macOS reports bytes; Linux reports KB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024 / 1024 if sys.platform == "darwin" else rss / 1024


def get_vault_candidates(collection, embed_fn, query_text: str,
                         top_k: int = 5, exclude_id: str = None) -> list[dict]:
    """Query ChromaDB for top-k notes similar to query_text."""
    try:
        count = collection.count()
    except Exception:
        return []
    if count == 0:
        return []

    query_vec = embed_fn(query_text)
    n = min(top_k + (1 if exclude_id else 0), count)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=n,
        include=["documents", "metadatas"],
    )
    candidates = []
    for doc_id, doc, meta in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0]
    ):
        if doc_id == exclude_id:
            continue
        candidates.append({"id": doc_id, "doc": doc, "tags": meta.get("tags", "")})
        if len(candidates) >= top_k:
            break
    return candidates


# ---------------------------------------------------------------------------
# Architecture runners
# ---------------------------------------------------------------------------

def run_single_agent(agent, note_text: str, collection, embed_fn, note_id: str) -> dict:
    """Condition A: one LLM call produces tags + summary + links."""
    t0 = time.perf_counter()
    candidates = get_vault_candidates(collection, embed_fn, note_text,
                                      top_k=5, exclude_id=note_id)
    result = agent.single_agent(note_text, candidates)
    elapsed = round(time.perf_counter() - t0, 2)
    return {
        **result,
        "elapsed_seconds": elapsed,
        "inference_calls": 1,
        "stages": [{
            "stage": "single_agent",
            "elapsed": elapsed,
            "prompt_tokens": agent.last_usage.get("prompt_tokens", 0),
            "completion_tokens": agent.last_usage.get("completion_tokens", 0),
        }],
    }


def run_multi_agent(agent, note_text: str, collection, embed_fn, note_id: str) -> dict:
    """Condition B: Tagger → Summarizer → Linker pipeline."""
    stages = []
    total_start = time.perf_counter()

    # Stage 1: Tagger
    t0 = time.perf_counter()
    tags = agent.tagger_agent(note_text)
    stages.append({
        "stage": "tagger",
        "elapsed": round(time.perf_counter() - t0, 2),
        "prompt_tokens": agent.last_usage.get("prompt_tokens", 0),
        "completion_tokens": agent.last_usage.get("completion_tokens", 0),
    })

    # Stage 2: Summarizer
    t0 = time.perf_counter()
    summary = agent.summarizer_agent(note_text, tags)
    stages.append({
        "stage": "summarizer",
        "elapsed": round(time.perf_counter() - t0, 2),
        "prompt_tokens": agent.last_usage.get("prompt_tokens", 0),
        "completion_tokens": agent.last_usage.get("completion_tokens", 0),
    })

    # Stage 3: Linker (queries vault with the distilled summary)
    t0 = time.perf_counter()
    candidates = get_vault_candidates(collection, embed_fn, summary,
                                      top_k=5, exclude_id=note_id)
    links = agent.linker_agent(summary, tags, candidates)
    stages.append({
        "stage": "linker",
        "elapsed": round(time.perf_counter() - t0, 2),
        "prompt_tokens": agent.last_usage.get("prompt_tokens", 0),
        "completion_tokens": agent.last_usage.get("completion_tokens", 0),
    })

    return {
        "tags": tags,
        "summary": summary,
        "links": links,
        "elapsed_seconds": round(time.perf_counter() - total_start, 2),
        "inference_calls": 3,
        "stages": stages,
    }


def run_multi_agent_overseer(agent, note_text: str, collection, embed_fn,
                              note_id: str, max_retries: int = 1) -> dict:
    """Condition C: multi-agent pipeline + Overseer quality control."""
    # Task routing: skip summarizer for very short notes (scheduling decision)
    note_word_count = len(note_text.split())
    skipped_summary = note_word_count < 50

    if skipped_summary:
        print(f"  Overseer routing: note is {note_word_count} words (<50) — skipping summarizer.")
        t0 = time.perf_counter()
        tags = agent.tagger_agent(note_text)
        elapsed_tagger = round(time.perf_counter() - t0, 2)
        stages = [{
            "stage": "tagger",
            "elapsed": elapsed_tagger,
            "prompt_tokens": agent.last_usage.get("prompt_tokens", 0),
            "completion_tokens": agent.last_usage.get("completion_tokens", 0),
        }]
        summary = " ".join(note_text.split()[:50])  # use raw text as stand-in
        candidates = get_vault_candidates(collection, embed_fn, summary,
                                          top_k=5, exclude_id=note_id)
        links = agent.linker_agent(summary, tags, candidates)
        stages.append({
            "stage": "linker",
            "elapsed": round(time.perf_counter() - t0 - elapsed_tagger, 2),
            "prompt_tokens": agent.last_usage.get("prompt_tokens", 0),
            "completion_tokens": agent.last_usage.get("completion_tokens", 0),
        })
        base_result = {
            "tags": tags, "summary": summary, "links": links,
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
            "inference_calls": 2, "stages": stages,
            "overseer_skipped_summary": True,
        }
    else:
        base_result = run_multi_agent(agent, note_text, collection, embed_fn, note_id)

    # Overseer quality control pass
    t0 = time.perf_counter()
    review = agent.overseer_agent(
        note_text, base_result["tags"], base_result["summary"],
        base_result.get("links", [])
    )
    overseer_elapsed = round(time.perf_counter() - t0, 2)
    base_result["stages"].append({
        "stage": "overseer",
        "elapsed": overseer_elapsed,
        "prompt_tokens": agent.last_usage.get("prompt_tokens", 0),
        "completion_tokens": agent.last_usage.get("completion_tokens", 0),
    })
    base_result["elapsed_seconds"] = round(
        base_result["elapsed_seconds"] + overseer_elapsed, 2
    )
    base_result["inference_calls"] += 1

    # Apply overseer revisions
    if review.get("revised_tags"):
        base_result["tags"] = review["revised_tags"]
    if review.get("revised_summary"):
        base_result["summary"] = review["revised_summary"]
    if review.get("revised_links"):
        base_result["links"] = review["revised_links"]

    # Retry once if overseer flagged a fundamental failure
    if review.get("retry_recommended") and max_retries > 0:
        print("  Overseer: retry recommended — re-running pipeline...")
        return run_multi_agent_overseer(
            agent, note_text, collection, embed_fn, note_id, max_retries - 1
        )

    base_result["overseer_review"] = review
    return base_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(vault_path: str, inputfile: str, tagger_prompt: str, summarizer_prompt: str,
         agent_model: str, embed_model: str, experiment: str, label: str,
         architecture: str, temperature: float):

    print(f"Vault Path: {vault_path}")
    if inputfile:
        print(f"Input File: {inputfile}")
    print(f"Architecture: {architecture} | Model: {agent_model} | Temp: {temperature}\n")

    # Load all prompts upfront
    prompts = {
        "tagger":       load_prompt("tagger", tagger_prompt),
        "summarizer":   load_prompt("summarizer", summarizer_prompt),
        "linker":       load_prompt("linker", "v1"),
        "single_agent": load_prompt("single_agent", "v1"),
        "overseer":     load_prompt("overseer", "v1"),
    }

    model_client = ollama.Client(host="http://localhost:11434")
    agent = Agent(model_client, prompts, model_type=agent_model, temperature=temperature)

    def embed(text: str) -> list[float]:
        return model_client.embeddings(prompt=text, model=embed_model)["embedding"]

    if inputfile:
        note_path = Path(vault_path) / inputfile
        note_text = note_path.read_text()
        print(f"=== {architecture}: {note_path} ===\n")

        try:
            collection = chromadb.PersistentClient().get_or_create_collection("second-brain")
        except Exception as e:
            print(f"Warning: ChromaDB unavailable ({e}). Linker will return no candidates.")
            collection = None

        rss_before = peak_rss_mb()

        if architecture == "single_agent":
            result = run_single_agent(agent, note_text, collection, embed, inputfile)
        elif architecture == "multi_agent":
            result = run_multi_agent(agent, note_text, collection, embed, inputfile)
        elif architecture == "multi_agent_overseer":
            result = run_multi_agent_overseer(agent, note_text, collection, embed, inputfile)
        else:
            raise ValueError(f"Unknown architecture: {architecture!r}")

        rss_delta = round(peak_rss_mb() - rss_before, 2)

        # --- Print outputs ---
        print(f"Tags:    {result.get('tags', [])}\n")
        print(f"Summary: {result.get('summary', '')}\n")
        links = result.get("links", [])
        if links:
            print("Links:")
            for lnk in links:
                print(f"  • {lnk.get('note_title', '?')} — {lnk.get('justification', '')}")
            print()
        if result.get("overseer_review"):
            rv = result["overseer_review"]
            status = "✓ approved" if rv.get("approved") else "✗ issues found"
            print(f"Overseer: {status} | issues: {rv.get('issues_found', [])}\n")
        total_tokens = sum(
            s.get("prompt_tokens", 0) + s.get("completion_tokens", 0)
            for s in result.get("stages", [])
        )
        print(f"Elapsed: {result['elapsed_seconds']}s | Calls: {result['inference_calls']} "
              f"| Tokens: {total_tokens} | ΔRSS: {rss_delta} MB\n")

        # --- Log experiment ---
        log_experiment(experiment, {
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "label":            label,
            "architecture":     architecture,
            "note_path":        inputfile,
            "agent_model":      agent_model,
            "embed_model":      embed_model,
            "tagger_prompt":    tagger_prompt,
            "summarizer_prompt": summarizer_prompt,
            "temperature":      temperature,
            "tags":             result.get("tags", []),
            "summary":          result.get("summary", ""),
            "links":            result.get("links", []),
            "overseer_review":  result.get("overseer_review"),
            "elapsed_seconds":  result["elapsed_seconds"],
            "inference_calls":  result["inference_calls"],
            "stages":           result.get("stages", []),
            "peak_rss_delta_mb": rss_delta,
        })
        print(f"Logged to experiments/{experiment}.jsonl")

    else:
        vault = Vault(vault_path, model_client, agent, model_type=embed_model)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, model_client, prompts: dict, model_type: str = "qwen3.5:9b",
                 temperature: float = 0.3):
        self.model_client = model_client
        self.model_type = model_type
        self.temperature = temperature
        self.last_usage: dict = {}
        self.tagger_prompt      = prompts.get("tagger", "")
        self.summarizer_prompt  = prompts.get("summarizer", "")
        self.linker_prompt      = prompts.get("linker", "")
        self.single_agent_prompt = prompts.get("single_agent", "")
        self.overseer_prompt    = prompts.get("overseer", "")

    def model_chat(self, messages: list[dict], output_format=None):
        response = self.model_client.chat(
            model=self.model_type,
            messages=messages,
            format=output_format,
            stream=False,
            think=False,
            options={"temperature": self.temperature},
        )
        # Extract token counts — Ollama returns these at the top level
        try:
            self.last_usage = {
                "prompt_tokens":     response.get("prompt_eval_count", 0) or 0,
                "completion_tokens": response.get("eval_count", 0) or 0,
            }
        except AttributeError:
            self.last_usage = {
                "prompt_tokens":     getattr(response, "prompt_eval_count", 0) or 0,
                "completion_tokens": getattr(response, "eval_count", 0) or 0,
            }
        return response

    def tagger_agent(self, note_text: str) -> list[str]:
        response = self.model_chat(
            messages=[
                {"role": "system", "content": self.tagger_prompt},
                {"role": "user",   "content": note_text},
            ],
            output_format="json",
        )
        return json.loads(response["message"]["content"])

    def summarizer_agent(self, note_text: str, tags: list[str]) -> str:
        user_message = (
            f"Note:\n{note_text}\n\n"
            f"Tags identified for this note: {', '.join(tags)}"
        )
        response = self.model_chat(
            messages=[
                {"role": "system", "content": self.summarizer_prompt},
                {"role": "user",   "content": user_message},
            ]
        )
        return response["message"]["content"]

    def linker_agent(self, summary: str, tags: list[str],
                     candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []
        candidates_text = "\n".join(
            f"- {c['id']}: {c['doc'][:300]}...\n  Tags: {c.get('tags', '')}"
            for c in candidates
        )
        user_message = (
            f"Current note summary: {summary}\n"
            f"Current note tags: {', '.join(tags)}\n\n"
            f"Candidate related notes:\n{candidates_text}"
        )
        response = self.model_chat(
            messages=[
                {"role": "system", "content": self.linker_prompt},
                {"role": "user",   "content": user_message},
            ],
            output_format="json",
        )
        result = json.loads(response["message"]["content"])
        return result if isinstance(result, list) else result.get("links", [])

    def single_agent(self, note_text: str, candidates: list[dict]) -> dict:
        candidates_text = (
            "\n".join(f"- {c['id']}: {c['doc'][:200]}..." for c in candidates)
            if candidates else "No vault notes available for linking."
        )
        user_message = (
            f"Note:\n{note_text}\n\n"
            f"Available vault notes for linking:\n{candidates_text}"
        )
        response = self.model_chat(
            messages=[
                {"role": "system", "content": self.single_agent_prompt},
                {"role": "user",   "content": user_message},
            ],
            output_format="json",
        )
        return json.loads(response["message"]["content"])

    def overseer_agent(self, note_text: str, tags: list[str],
                        summary: str, links: list[dict]) -> dict:
        user_message = (
            f"Original note:\n{note_text}\n\n"
            f"Agent outputs to review:\n"
            f"Tags: {tags}\n"
            f"Summary: {summary}\n"
            f"Links: {json.dumps(links, indent=2)}"
        )
        response = self.model_chat(
            messages=[
                {"role": "system", "content": self.overseer_prompt},
                {"role": "user",   "content": user_message},
            ],
            output_format="json",
        )
        return json.loads(response["message"]["content"])


# ---------------------------------------------------------------------------
# Vault (bulk indexing)
# ---------------------------------------------------------------------------

class Vault:
    def __init__(self, vault_path, model_client, agent, model_type="mxbai-embed-large"):
        self.vault_path = vault_path
        self.model_client = model_client
        self.vector_db = chromadb.PersistentClient()
        self.model_type = model_type
        # To drop and recreate the collection for a full reindex, uncomment:
        # try:
        #     self.vector_db.delete_collection(name="second-brain")
        #     print("Old collection dropped.")
        # except ValueError:
        #     pass
        self.collection = self.vector_db.get_or_create_collection("second-brain")
        self.agent = agent
        self.index_vault()

    def index_vault(self):
        existing_ids = set(self.collection.get()["ids"])
        metrics = {"indexed": 0, "skipped": 0}

        for filepath in Path(self.vault_path).rglob("*.md"):
            filename = str(filepath.relative_to(self.vault_path))
            if filename not in existing_ids:
                text = filepath.read_text()
                tags = self.agent.tagger_agent(text)
                summary = self.agent.summarizer_agent(text, tags)
                self.index_note(filename, summary, [{
                    "tags": ", ".join(tags),
                    "last_modified": filepath.stat().st_mtime,
                }])
                metrics["indexed"] += 1
                print(f"Indexed: {filename}")
            else:
                metrics["skipped"] += 1
        print(metrics)

    def index_note(self, filename: str, text: str, metadata_list=None):
        if filename in set(self.collection.get()["ids"]):
            return
        vector = self.model_client.embeddings(
            prompt=text, model=self.model_type
        )["embedding"]
        self.collection.add(
            ids=[filename],
            embeddings=[vector],
            documents=[text],
            metadatas=metadata_list,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Second Brain agent pipeline.")
    parser.add_argument("--vaultpath",          required=True,  help="Absolute path to your vault folder.")
    parser.add_argument("--inputfile",                          help="Note to process (relative to vaultpath). Omit to bulk-index.")
    parser.add_argument("--architecture",        default="multi_agent",
                        choices=["single_agent", "multi_agent", "multi_agent_overseer"],
                        help="Agent architecture to run (default: multi_agent).")
    parser.add_argument("--tagger-prompt",       default="v1",  help="Prompt version for tagger (e.g. v1, v2).")
    parser.add_argument("--summarizer-prompt",   default="v1",  help="Prompt version for summarizer.")
    parser.add_argument("--agent-model",         default="qwen3.5:9b", help="Ollama model for agents.")
    parser.add_argument("--embed-model",         default="mxbai-embed-large", help="Ollama model for embeddings.")
    parser.add_argument("--temperature",         default=0.3,   type=float, help="Sampling temperature (default 0.3).")
    parser.add_argument("--experiment",          default="baseline", help="Log category → experiments/<name>.jsonl.")
    parser.add_argument("--label",               default="",    help="Free-form run label.")
    args = parser.parse_args()

    main(
        args.vaultpath, args.inputfile,
        args.tagger_prompt, args.summarizer_prompt,
        args.agent_model, args.embed_model,
        args.experiment, args.label,
        args.architecture, args.temperature,
    )
