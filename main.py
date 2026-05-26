import argparse
import chromadb
import json
import ollama
import time
from datetime import datetime, timezone
from pathlib import Path


def load_prompt(category: str, name: str) -> str:
    return (Path(__file__).parent / "prompts" / category / f"{name}.txt").read_text()


def log_experiment(category: str, payload: dict):
    out_dir = Path(__file__).parent / "experiments"
    out_dir.mkdir(exist_ok=True)
    with (out_dir / f"{category}.jsonl").open("a") as f:
        f.write(json.dumps(payload) + "\n")


def main(vault_path: str, inputfile: str, tagger_prompt: str, summarizer_prompt: str,
         agent_model: str, embed_model: str, experiment: str, label: str):
    print("Vault Path: {}".format(vault_path))
    if inputfile:
        print("Input File: {}".format(inputfile))

    tagger_prompt_text = load_prompt("tagger", tagger_prompt)
    summarizer_prompt_text = load_prompt("summarizer", summarizer_prompt)

    model_client = ollama.Client(host="http://localhost:11434")
    agent = Agent(model_client, tagger_prompt_text, summarizer_prompt_text, model_type=agent_model)

    if inputfile:
        note_path = Path(vault_path) / inputfile
        test_note_text = note_path.read_text()
        print(f"\n=== Single-note test: {note_path} ===\n")

        start = time.perf_counter()

        tags = agent.tagger_agent(test_note_text)
        print(f"Tags: {tags}\n")

        summary = agent.summarizer_agent(test_note_text, tags)
        print(f"Summary: {summary}\n")

        embedding_response = model_client.embeddings(prompt=summary, model=embed_model)
        vec = embedding_response["embedding"]
        print(f"Embedding: dim={len(vec)}, first 5={vec[:5]}\n")

        elapsed = time.perf_counter() - start

        log_experiment(experiment, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "note_path": inputfile,
            "agent_model": agent_model,
            "embed_model": embed_model,
            "tagger_prompt": tagger_prompt,
            "summarizer_prompt": summarizer_prompt,
            "tags": tags,
            "summary": summary,
            "embedding_dim": len(vec),
            "embedding_preview": vec[:5],
            "elapsed_seconds": round(elapsed, 2),
        })
        print(f"Logged to experiments/{experiment}.jsonl")

    else:
        vault = Vault(vault_path, model_client, agent, model_type=embed_model)


class Agent:
    def __init__(self, model_client, tagger_prompt: str, summarizer_prompt: str, model_type='qwen3.5:9b'):
        self.model_client = model_client
        self.model_type = model_type
        self.tagger_prompt = tagger_prompt
        self.summarizer_prompt = summarizer_prompt

    def model_chat(self, messages: list[dict[str, str]], output_format=None):
        chat_response = self.model_client.chat(
            model=self.model_type,
            messages=messages,
            format=output_format,
            stream=False,
            think=False
        )
        return chat_response

    def tagger_agent(self, note_text: str) -> list[str]:
        response = self.model_chat(
            messages=[
                {"role": "system", "content": self.tagger_prompt},
                {"role": "user", "content": note_text}
            ],
            output_format='json'
        )
        raw = response["message"]["content"]
        tags = json.loads(raw)
        return tags

    def summarizer_agent(self, note_text: str, tags: list[str]) -> str:
        user_message = f"""Note:
        {note_text}

        Tags identified for this note: {', '.join(tags)}
        """

        response = self.model_chat(
            messages=[
                {"role": "system", "content": self.summarizer_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        return response["message"]["content"]


class Vault:
    def __init__(self, vault_path, model_client, agent, model_type="mxbai-embed-large"):
        self.vault_path = vault_path
        self.model_client = model_client
        self.vector_db = chromadb.PersistentClient()
        self.model_type = model_type
        # try:
        #     self.vector_db.delete_collection(name="second-brain")
        #     print("Old collection dropped successfully.")
        # except ValueError:
        #     print("Collection did not exist. Creating a fresh one.")
        self.collection = self.vector_db.get_or_create_collection("second-brain")
        self.agent = agent
        self.index_vault()

    def index_vault(self):
        existing_ids = set(self.collection.get()["ids"])  # what's already indexed
        index_metrics = {"indexed": 0, "skipped": 0}

        for filepath in Path(self.vault_path).rglob("*.md"):
            filename = str(filepath.relative_to(self.vault_path))    # will need to reconstruct full path to find file

            if filename not in existing_ids:
                text = filepath.read_text()
                tags = self.agent.tagger_agent(text)
                summary = self.agent.summarizer_agent(text, tags)
                metadata_list = [{
                    "tags": ', '.join(tags),
                    "last_modified": filepath.stat().st_mtime     # last modified time of the file
                }]
                self.index_note(filename, summary, metadata_list)
                index_metrics["indexed"] += 1
                print(f"Indexed: {filename}")
            else:
                index_metrics["skipped"] += 1
        print(index_metrics)

    def index_note(self, filename, text, metadata_list=None):
        existing_ids = set(self.collection.get()["ids"])  # what's already indexed
        if filename not in existing_ids:
            response = self.model_client.embeddings(
                prompt=text,
                model=self.model_type
            )
            vector = response['embedding']
            self.collection.add(
                ids=[filename],
                embeddings=[vector],
                documents=[text],
                metadatas=metadata_list
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Second Brain agent pipeline.")
    parser.add_argument("--vaultpath", type=str, help="Absolute path to your vault.", required=True)
    parser.add_argument("--inputfile", type=str, help="The note to tag, summarize, and link related notes to.")
    parser.add_argument("--tagger-prompt", default="v1", help="Prompt version for the tagger agent (e.g. v1, v2).")
    parser.add_argument("--summarizer-prompt", default="v1", help="Prompt version for the summarizer agent.")
    parser.add_argument("--agent-model", default="qwen3.5:9b", help="Ollama model for tag/summary agents.")
    parser.add_argument("--embed-model", default="mxbai-embed-large", help="Ollama model for embeddings.")
    parser.add_argument("--experiment", default="baseline", help="Category name; results append to experiments/<name>.jsonl.")
    parser.add_argument("--label", default="", help="Free-form note to identify this run.")
    args = parser.parse_args()
    main(args.vaultpath, args.inputfile, args.tagger_prompt, args.summarizer_prompt,
         args.agent_model, args.embed_model, args.experiment, args.label)
