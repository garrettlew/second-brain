import chromadb
from pathlib import Path

class Vault:
    """
    Manages the Obsidian vault and its vector database index.

    Handles indexing of mark down notes into a ChromaDB collection,
    including generating tags, summaries, and embeddings for each note.
    Also provides utilities for querying and updating notes with related links.
    """

    def __init__(self, vault_path, model_client, agent, model_type="mxbai-embed-large", chromadb_name="second-brain"):
        """
        Initializes the Vault and indexes any unindexed mark down notes.

        Args:
            vault_path (str): Absolute path to the Obsidian vault directory.
            model_client: The Ollama client used for embeddings and chat.
            agent: The Agent instance used for tagging and summarization.
            model_type (str): The embedding model to use. Defaults to 'mxbai-embed-large'.
        """

        self.vault_path = vault_path
        self.model_client = model_client
        self.vector_db = chromadb.PersistentClient()
        self.model_type = model_type
        self.chromadb_name = chromadb_name
        # try:
        #     self.vector_db.delete_collection(name="second-brain")
        #     print("Old collection dropped successfully.")
        # except ValueError:
        #     print("Collection did not exist. Creating a fresh one.")
        self.collection = self.vector_db.get_or_create_collection(
            name=self.chromadb_name,
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            }
        )
        self.agent = agent
        self.index_vault()


    def index_vault(self):
        """
        Scans the vault directory and indexes any mark down notes not yet in the collection.

        For each unindexed note, generates tags, a summary, and an embedding,
        then stores them in the vector database. Prints a summary of how many
        notes were indexed vs skipped.
        """

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
        """
        Generates an embedding for a note and adds it to the vector database.

        Skips indexing if the note is already present in the collection.

        Args:
            filename (str): The relative path of the note from the vault root, used as its ID.
            text (str): The text to embed and store (typically the note's summary).
            metadata_list (list[dict], optional): Metadata to store alongside the note,
                such as tags and last modified time.
        """

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
        """
         Appends a 'Related Notes' section to a note with links that don't already exist in it.

         Reads the note's current content and filters out any links already present
         before writing, preventing duplicate entries.

         Args:
             filename (str): The relative path of the note from the vault root.
             links (list[dict]): A list of links to append, each with keys:
                 - "id": the linked note's filename (used to construct the [[wikilink]])
                 - "reason": one sentence explaining the connection
         """

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


    def get_note_setup_from_vault(self, note_id):
        """
        Get the stored summary and tags for the input note from the shared Vault collection.
        In Vault.py, the ChromaDB document is the generated summary, and tags are stored in metadata.
        """

        result = self.collection.get(
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

    def query_related_notes_from_vault(self, note_id, query_embedding, final_k=3) -> list[dict]:
        """
        Query the shared Vault ChromaDB collection.

        We retrieve top 4 candidates, remove the input note itself if it appears,
        and keep the top 3 real candidate notes.

        Returns: [{
                    "note_title": candidate_id,
                    "summary": candidate_summary,
                    "tags": parse_tags(candidate_metadata),
                    "distance": results["distances"][0][i]
                }, ...]
        """

        total_notes = self.collection.count()

        if total_notes <= 1:
            return []

        raw_k = min(final_k + 1, total_notes)

        results = self.collection.query(
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
