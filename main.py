import argparse
import json
import ollama

from pydantic import BaseModel
from Vault import Vault


def main(vault_path: str, inputfile: str):
    print("Vault Path: {}".format(vault_path))
    model_client = ollama.Client(host="http://localhost:11434")
    agent = Agent(model_client)
    vault = Vault(vault_path, model_client, agent)
    if inputfile:
        print("Input File: {}".format(inputfile))

        current_note_results = vault.collection.get(
            ids=[inputfile],
            include=["embeddings", "documents", "metadatas"]
        )

        current_note_summary = current_note_results['documents'][0]
        current_note_metadata = current_note_results['metadatas'][0]
        current_note_tags = current_note_metadata.get('tags')

        candidate_note_results = vault.collection.query(
            query_embeddings=[current_note_results['embeddings'][0]],
            n_results=4  # ask for 4, discard the first (self)
        )

        # skip index 0 — that's the note itself
        # related = results["ids"][0][1:]
        print("IDs: {}".format(candidate_note_results['ids'][0]))
        print("Distances: {}".format(candidate_note_results['distances'][0]))
        candidate_note_id = candidate_note_results['ids'][0][3]
        candidate_note_summary = candidate_note_results['documents'][0][3]
        candidate_note_metadata = candidate_note_results['metadatas'][0][3]
        candidate_note_tags = candidate_note_metadata.get('tags')

        judgement = agent.linker_agent(current_note_tags, current_note_summary, candidate_note_tags,
                                       candidate_note_summary)

        print(judgement.relevant)
        print(judgement.reason)
        links = []
        if judgement.relevant:
            link = {"id": candidate_note_id, "reason": judgement.reason}
            links.append(link)
            vault.append_links_to_note(inputfile, links)
    else:
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


class Judgement(BaseModel):
    relevant: bool
    reason: str


class Agent:
    def __init__(self, model_client, model_type='qwen3.5:9b'):
        self.model_client = model_client
        self.model_type = model_type

    def model_chat(self, messages: list[dict[str, str]], output_format=None, think=False):
        chat_response = self.model_client.chat(
            model=self.model_type,
            messages=messages,
            format=output_format,
            stream=False,
            think=think
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

    def linker_agent(self, current_note_tags: list[str], current_note_summary: str, candidate_note_tags: list[str], candidate_note_summary: str) -> list[dict]:
        LINKER_SYSTEM_PROMPT = """
        You are a note linking agent. Your job is to decide if the provided candidate note is 
        genuinely relevant to link to the current note.

        Rules:
        - Only set relevant to true if a candidate has a meaningful conceptual connection to the current note
        - Reject candidates that are only superficially or tangentially related
        - For a kept candidate, write one sentence explaining the connection
        - You must respond with ONLY this exact JSON structure:
            {
                "relevant": true or false,
                "reason": "your one sentence reason here"
            }

        Relevant example output:
            {"relevant": true, "reason": "Both notes discuss attention mechanisms in neural networks"}
        
        Irrelevant example output:
            {"relevant": false, "reason": "Not related as the candidate note is about fence post embeddings while the current note is about the embeddings output of transformer encoders"}
        """

        user_message = f"""Current note summary:
            Tags: {current_note_tags}
            Summary: {current_note_summary}
            
            Candidate:
            Tags: {candidate_note_tags}
            Summary: {candidate_note_summary}
        
            Decide if the candidate is genuinely worth linking to the current note ("relevant": True) and give a reason why."""

        response = self.model_chat(
            messages=[
                {"role": "system", "content": LINKER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            output_format=Judgement.model_json_schema()
        )
        parsed = Judgement.model_validate_json(response.message.content)
        return parsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that greets you.")
    parser.add_argument("--vaultpath", type=str, help="Absolute path to your vault.", required=True)
    parser.add_argument("--inputfile", type=str, help="The note to tag, summarize, and link related notes to.")
    args = parser.parse_args()
    main(args.vaultpath, args.inputfile)
