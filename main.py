import argparse
import chromadb
import json
import ollama
from pathlib import Path


def main(vault_path: str, inputfile: str):
    print("Vault Path: {}".format(vault_path))
    if inputfile:
        print("Input File: {}".format(inputfile))

    model_client = ollama.Client(host="http://localhost:11434")
    agent = Agent(model_client)

    # test_filepath = Path('/Users/garrettlew/vault/example.md')
    # test_note_text = test_filepath.read_text()
    test_note_text = "The Talyllyn Railway is a narrow-gauge preserved railway in Wales running for 7.25 miles (11.67 km) from Tywyn on the Mid Wales coast to Nant Gwernol near the village of Abergynolwyn. The line was opened in 1866 to carry slate from the quarries at Bryn Eglwys to Tywyn, and was the first narrow-gauge railway in Britain authorised by act of Parliament to carry passengers using steam haulage. Despite severe under-investment, the line remained open, and on 14 May 1951 it became the first railway in the world to be operated as a heritage railway by volunteers. Since preservation, the railway has operated as a tourist attraction, significantly expanding its rolling stock through acquisition and an engineering programme to build new locomotives and carriages. The fictional Skarloey Railway, which formed part of the Railway Series of children's books by the Rev. W Awdry, was based on the Talyllyn Railway. The preservation of the line inspired the Ealing comedy film The Titfield Thunderbolt. "

    # 1. Generate tags
    tags = agent.tagger_agent(test_note_text)
    print(tags)
    # response = agent.model_chat([{"role": "user","content": "Hello world!"}])

    # 2. Use note + tags to generate summary
    summary = agent.summarizer_agent(test_note_text, tags)
    print(summary)

    # 3. Use summary to create embedding to store in vector database
    embedding_response = model_client.embeddings(
        prompt=summary,
        model="mxbai-embed-large"
    )
    print(embedding_response)

    # 4. Store note embedding into vector database
    vault = Vault(vault_path, model_client, agent)


class Agent:
    def __init__(self, model_client, model_type='qwen3.5:9b'):
        self.model_client = model_client
        self.model_type = model_type

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
        TAGGER_SYSTEM_PROMPT = """
        You are a note tagging agent. Your only job is to read a note and return relevant tags.

        Rules:
        - Return 3 tags
        - Tags should be lowercase, hyphenated (e.g. machine-learning, not Machine Learning)
        - Be specific but not overly narrow
        - Return ONLY a JSON list of strings [str, str, str], nothing else

        Example output:
        ["machine-learning", "meeting", "attention-mechanism"]
        """

        response = self.model_chat(
            messages=[
                {"role": "system", "content": TAGGER_SYSTEM_PROMPT},
                {"role": "user", "content": note_text}
            ],
            output_format='json'
        )
        print(response)
        raw = response["message"]["content"]
        tags = json.loads(raw)
        return tags

    def summarizer_agent(self, note_text: str, tags: list[str]) -> str:
        SUMMARIZER_SYSTEM_PROMPT = """
        You are a note summarization agent. Your job is to write a concise summary of a note.

        Rules:
        - Write at most 2-3 sentences
        - Focus on the core idea or insight of the note, not peripheral details
        - Use the provided tags as a guide for what the note is primarily about
        - Do not include opinions or evaluation of the content
        - Return ONLY the summary text, nothing else — no preamble, no labels

        Example output:
        The attention mechanism is the key idea in transformer models and it allows the model to weigh the relevance of different tokens. There are two types of attention: self-attention and cross-attention, and they're calculated using a scaled dot-product operation. The original 'Attention Is All You Need' paper is a key source.
        """

        user_message = f"""Note:
        {note_text}
    
        Tags identified for this note: {', '.join(tags)}
        """

        response = self.model_chat(
            messages=[
                {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )
        return response["message"]["content"]


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
    parser = argparse.ArgumentParser(description="A script that greets you.")
    parser.add_argument("--vaultpath", type=str, help="Absolute path to your vault.", required=True)
    parser.add_argument("--inputfile", type=str, help="The note to tag, summarize, and link related notes to.")
    args = parser.parse_args()
    main(args.vaultpath, args.inputfile)
