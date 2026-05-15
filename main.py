import chromadb
import json
import ollama
import pathlib


def main():
    model_client = ollama.Client(host="http://localhost:11434")
    agent = Agent(model_client)

    test_note_text = "The Talyllyn Railway is a narrow-gauge preserved railway in Wales running for 7.25 miles (11.67 km) from Tywyn on the Mid Wales coast to Nant Gwernol near the village of Abergynolwyn. The line was opened in 1866 to carry slate from the quarries at Bryn Eglwys to Tywyn, and was the first narrow-gauge railway in Britain authorised by act of Parliament to carry passengers using steam haulage. Despite severe under-investment, the line remained open, and on 14 May 1951 it became the first railway in the world to be operated as a heritage railway by volunteers. Since preservation, the railway has operated as a tourist attraction, significantly expanding its rolling stock through acquisition and an engineering programme to build new locomotives and carriages. The fictional Skarloey Railway, which formed part of the Railway Series of children's books by the Rev. W Awdry, was based on the Talyllyn Railway. The preservation of the line inspired the Ealing comedy film The Titfield Thunderbolt. "
    # test_tags_list = ['heritage-railway', 'preservation', 'steam-locomotive']

    # 1. Generate tags
    tags = agent.tagger_agent(test_note_text)
    print(tags)
    # response = agent.model_chat([{"role": "user","content": "Hello world!"}])

    # 2. Use note + tags to generate summary
    summary = agent.summarizer_agent(test_note_text, tags)
    print(summary)
    # test_summary = "This note details the history of the Talyllyn Railway, the first in the world to be preserved and operated by volunteers as a heritage railway. It highlights the line's transition into a tourist attraction through significant engineering efforts and the acquisition of new steam locomotives. Additionally, the text notes the railway's cultural impact, serving as the real-life model for the fictional Skarloey Railway and inspiring the film The Titfield Thunderbolt."

    # 3. Use summary to create embedding to store in vector database
    embedding_response = model_client.embeddings(
        prompt=summary,
        model="mxbai-embed-large"
    )
    print(embedding_response)

    # 4. Store note embedding into vector database
    # vault = Vault("/Users/garrettlew/vault")


class Agent:
    def __init__(self, model_client, model_type='qwen3.5:9b'):
        self.model_client = model_client
        self.model_type = model_type

    def model_chat(self, messages: list[dict[str, str]], output_format='json'):
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
        - Return ONLY a list of strings, nothing else

        Example output:
        ["machine-learning", "meeting", "attention-mechanism"]
        """

        response = self.model_chat(
            messages=[
                {"role": "system", "content": TAGGER_SYSTEM_PROMPT},
                {"role": "user", "content": note_text}
            ]
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
        This note covers the attention mechanism in transformer models and how it allows the model to weigh the relevance of different tokens. It explains the difference between self-attention and cross-attention, with a focus on the scaled dot-product operation. The note also references the original 'Attention Is All You Need' paper as a key source.
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
    def __init__(self, vault_path, model_client, model_type="mxbai-embed-large"):
        self.vault_path = vault_path
        self.model_client = model_client
        self.vector_db = chromadb.PersistentClient(path=vault_path)
        self.model_type = "mxbai-embed-large"
        # try:
        #     self.vector_db.delete_collection(name="second-brain")
        #     print("Old collection dropped successfully.")
        # except ValueError:
        #     print("Collection did not exist. Creating a fresh one.")
        self.collection = self.vector_db.get_or_create_collection("second-brain")
        self.index_vault()


    def index_vault(self):
        existing_ids = set(self.collection.get()["ids"])  # what's already indexed

        for filepath in pathlib.Path(self.vault_path).rglob("*.md"):
            filename = str(filepath)

            if filename not in existing_ids:
                text = filepath.read_text()
                self.index_note(filename, text)
                print(f"Indexed: {filename}")


    def index_note(self, filename, text):
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
                documents=[text]
            )


if __name__ == "__main__":
    main()
