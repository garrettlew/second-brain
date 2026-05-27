import chromadb
from pathlib import Path

class Vault:
    def __init__(self, vault_path, model_client, agent, model_type="mxbai-embed-large"):
        self.vault_path = vault_path
        self.model_client = model_client
        self.vector_db = chromadb.PersistentClient()
        self.model_type = "mxbai-embed-large"
        # try:
        #     self.vector_db.delete_collection(name="second-brain")
        #     print("Old collection dropped successfully.")
        # except ValueError:
        #     print("Collection did not exist. Creating a fresh one.")
        self.collection = self.vector_db.get_or_create_collection(
            name="second-brain",
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            }
        )
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

    def append_links_to_note(self, filename: str, links: list[dict]):
        filepath = Path(self.vault_path) / filename
        content = filepath.read_text()

        new_links = [
            link for link in links
            if f"[[{link["id"]}]]" not in content
        ]

        if new_links:
            with filepath.open("a") as f:
                f.write("\n\n## Related Notes\n")
                for link in new_links:
                    f.write(f"- [[{link["id"]}]] — {link["reason"]}\n")
